// ============================================================================
//  infer_mlp.cpp  --  Homomorphic MLP inference (OpenFHE / CKKS)
//
//  Generalized version of lib/cpp/matrix_inference_sample.cpp: that file
//  hardcodes exactly 3 Linear layers (W1/W2/W3). exp03-mnist's grid search
//  trains MLPs with a variable number of hidden layers (e.g. layers=[784,10]
//  or [784,784,10] or [784,256,128,64,10]), so this loads however many
//  Wi_diag/bi pairs the model JSON actually has.
//
//  Forward pass (all on encrypted data), for N Linear layers:
//       z_i = W_i @ x_{i-1} + b_i
//       x_i = z_i^2               (all but the last layer)
//       logits = z_N              (last layer: no activation)
//
//  Matrix-vector product via Halevi-Shoup diagonal packing:
//       W @ x  =  sum_i  diag_i(W)  *  rotate(x, i)
//
//  Multiplicative depth: each layer's matVec+bias-add costs 1 level (pt*ct
//  mult); every layer but the last is followed by a squaring (ct*ct, 1
//  level). exact = 2*N - 1; we set depth = 2*N for headroom, matching the
//  depth=6 (exact 5, +1 headroom) used for the fixed 3-layer case.
//
//  Model is loaded from ckks_model.json produced by export_mdl.py.
//  Input is a plaintext feature vector (client normalizes and passes it in);
//  the binary encrypts it, runs inference, decrypts logits, and prints one
//  line per sample.
//
//  Usage:
//    infer_mlp <model.json> <features.txt> <key_file>
//
//    model.json    exported by export_mdl.py
//    features.txt  exported by export_features.py  (one sample per line: key f0 f1 ...)
//    key_file      one sample key per line; only these samples are processed
// ============================================================================

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include "openfhe.h"

using namespace lbcrypto;

struct MlpModel {
    int input_dim, num_classes, packed_dim;
    std::vector<std::string> classes;
    std::vector<double> mean, stdv;
    std::vector<std::vector<std::vector<double>>> W_diag;  // one entry per layer
    std::vector<std::vector<double>>              b;       // one entry per layer
};

inline MlpModel loadMlpModel(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    nlohmann::json j = nlohmann::json::parse(f);

    MlpModel m;
    m.input_dim   = j["input_dim"];
    m.num_classes = j["num_classes"];
    m.packed_dim  = j["packed_dim"];
    m.classes     = j["classes"].get<std::vector<std::string>>();
    m.mean        = j["mean"].get<std::vector<double>>();
    m.stdv        = j["std"].get<std::vector<double>>();

    for (int i = 1; j.contains("W" + std::to_string(i) + "_diag"); ++i) {
        m.W_diag.push_back(j["W" + std::to_string(i) + "_diag"].get<std::vector<std::vector<double>>>());
        m.b.push_back(j["b" + std::to_string(i)].get<std::vector<double>>());
    }
    if (m.W_diag.empty()) throw std::runtime_error("model has no layers (W1_diag missing): " + path);
    return m;
}

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
        Plaintext pt              = cc->MakeCKKSPackedPlaintext(diags[i]);
        Ciphertext<DCRTPoly> xr   = (i == 0) ? x : cc->EvalRotate(x, i);
        Ciphertext<DCRTPoly> term = cc->EvalMult(xr, pt);
        acc = (i == 0) ? term : cc->EvalAdd(acc, term);
    }
    return acc;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: infer_mlp <model.json> <features.txt> <key_file>\n"
                  << "  model.json    exported by 'export_mdl.py'\n"
                  << "  features.txt  exported by 'export_features.py'\n"
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
    MlpModel m = loadMlpModel(modelPath);
    const int dim        = m.packed_dim;
    const int numLayers  = static_cast<int>(m.W_diag.size());
    const int depth      = 2 * numLayers;
    std::cerr << "Loaded model: " << numLayers << " layer(s), packed_dim=" << dim
               << ", depth=" << depth << "\n";

    // ---- 1. CKKS context ---------------------------------------------------
    CCParams<CryptoContextCKKSRNS> params;
    params.SetMultiplicativeDepth(depth);
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

        Ciphertext<DCRTPoly> x = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(feat));

        // Server: homomorphic forward pass, one iteration per layer
        for (int layer = 0; layer < numLayers; ++layer) {
            Plaintext pt_b = cc->MakeCKKSPackedPlaintext(m.b[layer]);
            Ciphertext<DCRTPoly> z = matVec(cc, m.W_diag[layer], x, dim);
            z = cc->EvalAdd(z, pt_b);
            x = (layer < numLayers - 1) ? cc->EvalMult(z, z) : z;  // squared activation, or raw logits on the last layer
        }

        // Client: decrypt + output
        Plaintext res;
        cc->Decrypt(keys.secretKey, x, &res);
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
