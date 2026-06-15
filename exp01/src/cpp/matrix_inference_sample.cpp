// ============================================================================
//  fhe_server.cpp  --  Homomorphic MLP inference (OpenFHE / CKKS)
//
//  Forward pass (all on encrypted data):
//       z1 = W1 @ x  + b1
//       h1 = z1^2
//       z2 = W2 @ h1 + b2
//       h2 = z2^2
//       z3 = W3 @ h2 + b3    <- encrypted logits, sent to client
//
//  Matrix-vector product via Halevi-Shoup diagonal packing:
//       W @ x  =  sum_i  diag_i(W)  *  rotate(x, i)
//
//  Multiplicative depth:
//       fc1(pt*ct=1) + sq(ct*ct=1) + fc2(1) + sq(1) + fc3(1) = 5  -> depth 6 with headroom
//
//  Model is loaded from  ckks_model.json  produced by  export_ckks.py.
//  Input is a plaintext feature vector (client normalizes and passes it in);
//  the binary encrypts it, runs inference, decrypts logits, and prints one line per sample.
//
//  Usage:
//    fhe_dps_test <model.json> <features.txt> <key_file>
//
//    model.json    exported by export_ckks.py
//    features.txt  exported by export_features.py  (one sample per line: key f0 f1 ...)
//    key_file      one sample key per line; only these samples are processed
// ============================================================================

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <string>
#include "model.h"
#include "openfhe.h"

using namespace lbcrypto;

// ---------------------------------------------------------------------------
// Halevi-Shoup diagonal matrix-vector product:  out = W . x
static Ciphertext<DCRTPoly> matVec(
        const CryptoContext<DCRTPoly>& cc,
        const std::vector<std::vector<double>>& diags,
        const Ciphertext<DCRTPoly>& x,
        int dim)
{
    Ciphertext<DCRTPoly> acc;
    for (int i = 0; i < dim; ++i) {
        Plaintext pt               = cc->MakeCKKSPackedPlaintext(diags[i]);
        Ciphertext<DCRTPoly> xr   = (i == 0) ? x : cc->EvalRotate(x, i);
        Ciphertext<DCRTPoly> term = cc->EvalMult(xr, pt);
        acc = (i == 0) ? term : cc->EvalAdd(acc, term);
    }
    return acc;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: fhe_dps_test <model.json> <features.txt> <key_file>\n"
                  << "  model.json    exported by export_ckks.py\n"
                  << "  features.txt  exported by export_features.py\n"
                  << "  key_file      one sample key per line\n";
        return 1;
    }
    std::string modelPath   = argv[1];
    std::string featPath    = argv[2];
    std::string keyFilePath = argv[3];

    // ---- 0. Load keys to process -------------------------------------------
    std::unordered_set<std::string> targetKeys;
    {
        std::ifstream kf(keyFilePath);
        if (!kf) { std::cerr << "cannot open key file: " << keyFilePath << "\n"; return 1; }
        std::string k;
        while (std::getline(kf, k))
            if (!k.empty()) targetKeys.insert(k);
    }
    if (targetKeys.empty()) return 0;

    std::cout << std::setprecision(10);
    Model m = loadModel(modelPath);
    const int dim = m.packed_dim;

    // ---- 1. CKKS context ---------------------------------------------------
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(6);
    params.SetScalingModSize(50);
    params.SetBatchSize(dim);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<int> rots;
    for (int r = 1; r < dim; ++r) rots.push_back(r);
    cc->EvalRotateKeyGen(keys.secretKey, rots);

    // ---- 2. Iterate features, process matching keys ------------------------
    std::ifstream f(featPath);
    if (!f) { std::cerr << "cannot open features: " << featPath << "\n"; return 1; }

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (targetKeys.find(key) == targetKeys.end()) continue;

        // Client: read + normalize + encrypt
        std::vector<double> feat(dim, 0.0);
        for (int i = 0; i < m.input_dim; ++i) ss >> feat[i];
        for (int i = 0; i < m.input_dim; ++i)
            feat[i] = (feat[i] - m.mean[i]) / m.stdv[i];

        Ciphertext<DCRTPoly> cx = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(feat));

        // Server: homomorphic forward pass
        Plaintext pt_b1 = cc->MakeCKKSPackedPlaintext(m.b1);
        Plaintext pt_b2 = cc->MakeCKKSPackedPlaintext(m.b2);
        Plaintext pt_b3 = cc->MakeCKKSPackedPlaintext(m.b3);

        Ciphertext<DCRTPoly> z1 = matVec(cc, m.W1_diag, cx, dim);
        z1 = cc->EvalAdd(z1, pt_b1);
        Ciphertext<DCRTPoly> h1 = cc->EvalMult(z1, z1);

        Ciphertext<DCRTPoly> z2 = matVec(cc, m.W2_diag, h1, dim);
        z2 = cc->EvalAdd(z2, pt_b2);
        Ciphertext<DCRTPoly> h2 = cc->EvalMult(z2, z2);

        Ciphertext<DCRTPoly> z3 = matVec(cc, m.W3_diag, h2, dim);
        z3 = cc->EvalAdd(z3, pt_b3);

        // Client: decrypt + output
        Plaintext res;
        cc->Decrypt(keys.secretKey, z3, &res);
        res->SetLength(m.num_classes);
        std::vector<double> logits = res->GetRealPackedValue();

        std::cout << key;
        for (int i = 0; i < m.num_classes; ++i)
            std::cout << "  " << logits[i];
        std::cout << "\n";
        std::cout.flush();
    }

    return 0;
}
