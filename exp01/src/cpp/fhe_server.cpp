// ============================================================================
//  fhe_server.cpp  --  Homomorphic acoustic-scene classifier (OpenFHE / CKKS)
//
//  Computes, entirely on encrypted data:
//       z1 = W1 . x + b1
//       h  = z1 ^ 2                 (square activation; cheap under CKKS)
//       z2 = W2 . h  + b2           (encrypted logits -> sent to client)
//
//  Matrix-vector products use Halevi-Shoup DIAGONAL packing:
//       W.x = sum_i  diag_i  *  rotate(x, i)
//  so the whole layer needs one packed input ciphertext and `dim` rotations.
//
//  Multiplicative depth: matmul(1) + square(1) + matmul(1) = 3  -> no bootstrap.
//
//  This single binary plays both "client crypto setup" and "server compute"
//  for demo simplicity. In a real split you would serialize keys/ciphertexts
//  across the network; the compute section is exactly what a real server runs.
// ============================================================================

#include "openfhe.h"
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using namespace lbcrypto;

// ---- tiny JSON reader (depends only on the fields train.py writes) ---------
//  We avoid a JSON library dependency: the file is machine-generated and flat,
//  so we parse the specific keys we need with a minimal scanner.
#include "json_mini.h"   // defines struct Model + loadModel()

// Diagonal-packed matrix-vector product:  out = M . x
//   M given as `dim` diagonals (each length dim), x is a packed ciphertext.
//   result = sum_i  diag_i (plaintext) * rotate(x, i)
static Ciphertext<DCRTPoly> matVecDiag(
        const CryptoContext<DCRTPoly>& cc,
        const std::vector<std::vector<double>>& diags,
        const Ciphertext<DCRTPoly>& x, int dim) {

    Ciphertext<DCRTPoly> acc;
    for (int i = 0; i < dim; ++i) {
        Plaintext d = cc->MakeCKKSPackedPlaintext(diags[i]);
        auto xr   = (i == 0) ? x : cc->EvalRotate(x, i);
        auto term = cc->EvalMult(xr, d);
        acc = (i == 0) ? term : cc->EvalAdd(acc, term);
    }
    return acc;   // diagonal encoding yields M.x directly in the slots
}

int main(int argc, char** argv) {
    std::string modelPath = (argc > 1) ? argv[1] : "../model/model.json";
    std::string featPath  = (argc > 2) ? argv[2] : "../model/features.txt";

    Model m = loadModel(modelPath);   // from json_mini.h
    const int dim = m.packed_dim;
    std::cout << "Loaded model: " << m.num_classes << " classes, packed_dim "
              << dim << "\n";

    // ---- 1. CKKS context -------------------------------------------------
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(4);          // 3 used + 1 headroom
    params.SetScalingModSize(50);
    params.SetBatchSize(dim);
    params.SetScalingTechnique(FLEXIBLEAUTO);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // rotation keys 1..dim-1: matVecDiag rotates x by each diagonal index i
    std::vector<int> rots;
    for (int r = 1; r < dim; ++r) rots.push_back(r);   // demo: all rotations
    cc->EvalRotateKeyGen(keys.secretKey, rots);

    // ---- 2. Client side: read normalized feature vector, encrypt ---------
    //   features.txt is one line of `input_dim` doubles (already MFCC-extracted
    //   by the Python client; normalization applied below to mirror training).
    std::vector<double> feat(dim, 0.0);
    {
        std::ifstream f(featPath);
        if (!f) { std::cerr << "missing " << featPath << "\n"; return 1; }
        for (int i = 0; i < m.input_dim; ++i) f >> feat[i];
    }
    // normalize with the SAME mean/std used in training
    for (int i = 0; i < dim; ++i)
        feat[i] = (feat[i] - m.mean[i]) / m.stdv[i];

    Plaintext px = cc->MakeCKKSPackedPlaintext(feat);
    auto cx = cc->Encrypt(keys.publicKey, px);

    // ---- 3. SERVER side: homomorphic inference --------------------------
    //  Layer 1:  z1 = W1 . x + b1
    auto z1 = matVecDiag(cc, m.W1_diag, cx, dim);
    z1 = cc->EvalAdd(z1, cc->MakeCKKSPackedPlaintext(m.b1));

    //  square activation:  h = z1^2
    auto h = cc->EvalMult(z1, z1);

    //  Layer 2:  z2 = W2 . h + b2
    auto z2 = matVecDiag(cc, m.W2_diag, h, dim);
    z2 = cc->EvalAdd(z2, cc->MakeCKKSPackedPlaintext(m.b2));

    // ---- 4. Client side: decrypt logits, argmax -------------------------
    Plaintext res;
    cc->Decrypt(keys.secretKey, z2, &res);
    res->SetLength(m.num_classes);
    auto logits = res->GetRealPackedValue();

    int best = 0;
    for (int i = 1; i < m.num_classes; ++i)
        if (logits[i] > logits[best]) best = i;

    std::cout << "\nEncrypted-inference logits:\n";
    for (int i = 0; i < m.num_classes; ++i)
        std::cout << "  " << m.classes[i] << ": " << logits[i] << "\n";
    std::cout << "\nPredicted scene: " << m.classes[best] << "\n";
    return 0;
}
