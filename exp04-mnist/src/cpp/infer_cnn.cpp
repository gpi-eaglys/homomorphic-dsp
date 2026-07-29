// ============================================================================
//  infer_cnn.cpp  --  Homomorphic CNN inference (OpenFHE / CKKS)
//
//  Companion to exp03-mnist/src/cpp/infer_mlp.cpp, for CNNs (Conv2d + AvgPool2d +
//  Quadratic activation) instead of plain Linear stacks. Convolution/pooling never
//  repacks the ciphertext: each channel is one packed_dim-length ciphertext holding a
//  *fixed* w_phys x w_phys physical buffer for the whole network. A logical pixel
//  (row,col) at the model's current `stride` (doubling per pool) lives at physical slot
//  row*stride*w_phys + col*stride — see export_mdl.py's module docstring for the full
//  rationale (this trick, and why AvgPool2d's 1/4 scale is folded into the *next*
//  linear layer's weights rather than applied on its own).
//
//  Forward pass (all on encrypted data), per conv layer:
//    for each (c_in, ky, kx) term: rotate x[c_in] ONCE (reused across every c_out),
//    multiply per c_out by a plaintext that already has export_mdl.py's SAME-padding
//    mask and any pending pool scale folded into the weight; sum into z[c_out]; add
//    bias (scalar, broadcasts to every slot); square (Quadratic activation); then sum
//    4 fixed rotations for AvgPool2d (no scalar multiply -- the 1/4 was already folded
//    into the NEXT layer's weights by export_mdl.py).
//
//  Flatten -> FC1 is still one ciphertext per channel: FC1 is evaluated as one
//  Halevi-Shoup matVec per channel (only over the diagonals export_mdl.py found to be
//  actually nonzero), summed across channels. Any remaining dense FC layers reuse
//  lib/cpp/halevi_shoup.h's matVec() verbatim, exactly like infer_mlp.cpp.
//
//  Multiplicative depth: 1 level per conv layer + 1 level per FC (matVec) layer + 1
//  level per Quadratic squaring (after every conv layer and every FC layer except the
//  final output layer); pooling costs 0 levels. See computeDepth() below.
//
//  Usage:
//    infer_cnn <model.json> <features.txt> <id_file> <output.txt>
//
//    model.json    exported by export_mdl.py
//    features.txt  exported by exp03-mnist's export_features.py (shared, architecture-agnostic)
//    id_file       one sample id per line; only these samples are processed
//    output.txt    results are written here (one line per sample: id logit_0 ... logit_9)
// ============================================================================

#include <chrono>
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
#include "halevi_shoup.h"

using namespace lbcrypto;

struct ConvTerm {
    int c_in_idx;
    int shift;
    std::vector<std::vector<double>> plaintexts;  // [c_out][packed_dim]
};

struct ConvLayer {
    int c_in, c_out, kernel_size;
    int h_active, w_active, pool_stride_before;
    std::vector<ConvTerm> terms;
    std::vector<double> bias;  // [c_out], unscaled
};

struct Fc1Diagonal {
    int shift;
    std::vector<double> values;
};

struct Fc1Channel {
    int c_in_idx;
    std::vector<Fc1Diagonal> diagonals;  // only the nonzero ones (export_mdl.py measured this)
};

struct Fc1Layer {
    int in_channels, h_active, w_active, out_features;
    std::vector<Fc1Channel> channels;
    std::vector<double> bias;  // padded to packed_dim, unscaled
};

struct FcDenseLayer {
    std::vector<std::vector<double>> W_diag;  // dense, packed_dim entries
    std::vector<double> bias;
};

struct CnnModel {
    int packed_dim, w_phys, num_classes;
    std::vector<std::string> classes;
    std::vector<double> mean, stdv;
    std::vector<int> rotations;
    std::vector<ConvLayer> conv_layers;
    Fc1Layer fc1;
    std::vector<FcDenseLayer> fc_dense;
};

inline CnnModel loadCnnModel(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open model: " + path);
    nlohmann::json j = nlohmann::json::parse(f);

    CnnModel m;
    m.packed_dim  = j["packed_dim"];
    m.w_phys      = j["w_phys"];
    m.num_classes = j["num_classes"];
    m.classes     = j["classes"].get<std::vector<std::string>>();
    m.mean        = j["mean"].get<std::vector<double>>();
    m.stdv        = j["std"].get<std::vector<double>>();
    m.rotations   = j["rotations"].get<std::vector<int>>();

    for (const auto& jc : j["conv_layers"]) {
        ConvLayer layer;
        layer.c_in               = jc["c_in"];
        layer.c_out              = jc["c_out"];
        layer.kernel_size        = jc["kernel_size"];
        layer.h_active           = jc["h_active"];
        layer.w_active           = jc["w_active"];
        layer.pool_stride_before = jc["pool_stride_before"];
        layer.bias               = jc["bias"].get<std::vector<double>>();
        for (const auto& jt : jc["terms"]) {
            ConvTerm term;
            term.c_in_idx  = jt["c_in_idx"];
            term.shift     = jt["shift"];
            term.plaintexts = jt["plaintexts"].get<std::vector<std::vector<double>>>();
            layer.terms.push_back(std::move(term));
        }
        m.conv_layers.push_back(std::move(layer));
    }

    const auto& jf1 = j["fc1"];
    m.fc1.in_channels  = jf1["in_channels"];
    m.fc1.h_active     = jf1["h_active"];
    m.fc1.w_active     = jf1["w_active"];
    m.fc1.out_features = jf1["out_features"];
    m.fc1.bias         = jf1["bias"].get<std::vector<double>>();
    for (const auto& jch : jf1["channels"]) {
        Fc1Channel ch;
        ch.c_in_idx = jch["c_in_idx"];
        for (const auto& jd : jch["diagonals"]) {
            Fc1Diagonal d;
            d.shift  = jd["shift"];
            d.values = jd["values"].get<std::vector<double>>();
            ch.diagonals.push_back(std::move(d));
        }
        m.fc1.channels.push_back(std::move(ch));
    }

    for (const auto& jl : j["fc_dense"]) {
        FcDenseLayer layer;
        layer.W_diag = jl["W_diag"].get<std::vector<std::vector<double>>>();
        layer.bias   = jl["bias"].get<std::vector<double>>();
        m.fc_dense.push_back(std::move(layer));
    }

    return m;
}

// Exact multiplicative depth: 1 level per conv layer + 1 per FC (matVec) layer, plus 1
// per Quadratic squaring -- every conv layer, FC1 (iff more linear layers follow), and
// every dense FC layer except the last (the final output layer is never activated).
inline int computeDepth(const CnnModel& m) {
    int numConv     = static_cast<int>(m.conv_layers.size());
    int numFcLinear = 1 + static_cast<int>(m.fc_dense.size());  // fc1 + dense layers
    int numMults    = numConv + numFcLinear;

    int numSquarings = numConv;
    if (!m.fc_dense.empty()) {
        numSquarings += 1;                                   // fc1 is activated
        numSquarings += static_cast<int>(m.fc_dense.size()) - 1;  // all dense but the last
    }
    return numMults + numSquarings;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: infer_cnn <model.json> <features.txt> <id_file> <output.txt>\n"
                  << "  model.json    exported by 'export_mdl.py'\n"
                  << "  features.txt  exported by exp03-mnist's 'export_features.py'\n"
                  << "  id_file       one sample id per line\n"
                  << "  output.txt    results are written here\n";
        return 1;
    }
    std::string modelPath  = argv[1];
    std::string featPath   = argv[2];
    std::string idFilePath = argv[3];
    std::string outPath    = argv[4];

    // ---- 0. Load ids to process ---------------------------------------------
    std::unordered_set<std::string> targetIds;
    {
        std::ifstream idf(idFilePath);
        if (!idf) { std::cerr << "cannot open id file: " << idFilePath << "\n"; return 1; }
        std::string id;
        while (std::getline(idf, id))
            if (!id.empty()) targetIds.insert(id);
    }
    if (targetIds.empty()) return 0;
    std::cerr << "Running " << targetIds.size() << " inference(s)\n";

    std::ofstream out(outPath);
    if (!out) { std::cerr << "cannot open output file: " << outPath << "\n"; return 1; }
    out << std::setprecision(10);

    CnnModel m = loadCnnModel(modelPath);
    const int dim       = m.packed_dim;
    const int inputDim  = m.w_phys * m.w_phys;
    const int depthExact = computeDepth(m);
    const int depth      = depthExact + 1;  // headroom, matching infer_mlp.cpp's convention
    std::cerr << "Loaded model: " << m.conv_layers.size() << " conv layer(s), "
              << m.fc_dense.size() << " dense FC layer(s), packed_dim=" << dim
              << ", depth=" << depth << " (exact=" << depthExact << ")\n";

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
    cc->EvalRotateKeyGen(keys.secretKey, m.rotations);  // precomputed by export_mdl.py -- no recomputation here

    // ---- 2. Iterate features, process matching ids --------------------------
    std::ifstream f(featPath);
    if (!f) { std::cerr << "cannot open features: " << featPath << "\n"; return 1; }

    const size_t total = targetIds.size();
    size_t done = 0;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string id;
        ss >> id;
        if (targetIds.find(id) == targetIds.end()) continue;

        auto t0 = std::chrono::steady_clock::now();

        // Client: read + normalize + encrypt (only the first inputDim slots are real
        // pixels -- the rest of the packed_dim buffer stays zero, same as infer_mlp.cpp).
        std::vector<double> feat(dim, 0.0);
        for (int i = 0; i < inputDim; ++i) ss >> feat[i];
        for (int i = 0; i < inputDim; ++i)
            feat[i] = (feat[i] - m.mean[i]) / m.stdv[i];

        std::vector<Ciphertext<DCRTPoly>> x;
        x.push_back(cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(feat)));

        // Server: homomorphic forward pass -- conv/activation/pool stack
        for (const auto& conv : m.conv_layers) {
            std::vector<Ciphertext<DCRTPoly>> z(conv.c_out);
            std::vector<bool> started(conv.c_out, false);

            for (const auto& term : conv.terms) {
                Ciphertext<DCRTPoly> rotated = (term.shift == 0) ? x[term.c_in_idx]
                                                                  : cc->EvalRotate(x[term.c_in_idx], term.shift);
                for (int co = 0; co < conv.c_out; ++co) {
                    Plaintext pt = cc->MakeCKKSPackedPlaintext(term.plaintexts[co]);
                    Ciphertext<DCRTPoly> contrib = cc->EvalMult(rotated, pt);
                    z[co] = started[co] ? cc->EvalAdd(z[co], contrib) : contrib;
                    started[co] = true;
                }
            }

            const int s = conv.pool_stride_before;
            std::vector<Ciphertext<DCRTPoly>> pooled(conv.c_out);
            for (int co = 0; co < conv.c_out; ++co) {
                z[co] = cc->EvalAdd(z[co], conv.bias[co]);  // bias broadcasts to every slot
                z[co] = cc->EvalMult(z[co], z[co]);         // Quadratic activation

                Ciphertext<DCRTPoly> p = z[co];
                p = cc->EvalAdd(p, cc->EvalRotate(z[co], s));
                p = cc->EvalAdd(p, cc->EvalRotate(z[co], s * m.w_phys));
                p = cc->EvalAdd(p, cc->EvalRotate(z[co], s * m.w_phys + s));
                pooled[co] = p;  // unscaled -- 1/4 was folded into the NEXT layer's weights
            }
            x = std::move(pooled);
        }

        // FC1: per-channel sparse-diagonal matVec, summed across channels
        Ciphertext<DCRTPoly> acc;
        bool accStarted = false;
        for (const auto& ch : m.fc1.channels) {
            for (const auto& d : ch.diagonals) {
                Ciphertext<DCRTPoly> rotated = (d.shift == 0) ? x[ch.c_in_idx]
                                                               : cc->EvalRotate(x[ch.c_in_idx], d.shift);
                Plaintext pt = cc->MakeCKKSPackedPlaintext(d.values);
                Ciphertext<DCRTPoly> contrib = cc->EvalMult(rotated, pt);
                acc = accStarted ? cc->EvalAdd(acc, contrib) : contrib;
                accStarted = true;
            }
        }
        Plaintext pt_fc1_b = cc->MakeCKKSPackedPlaintext(m.fc1.bias);
        acc = cc->EvalAdd(acc, pt_fc1_b);

        // Remaining dense FC layers (exp03-style, reusing halevi_shoup.h's matVec())
        Ciphertext<DCRTPoly> cur = acc;
        if (!m.fc_dense.empty()) {
            cur = cc->EvalMult(cur, cur);  // FC1 is a hidden layer -> activated
            for (size_t i = 0; i < m.fc_dense.size(); ++i) {
                Ciphertext<DCRTPoly> z = matVec(cc, m.fc_dense[i].W_diag, cur, dim);
                Plaintext pt_dense_b = cc->MakeCKKSPackedPlaintext(m.fc_dense[i].bias);
                z = cc->EvalAdd(z, pt_dense_b);
                if (i + 1 < m.fc_dense.size())
                    z = cc->EvalMult(z, z);  // activation on every dense layer except the last
                cur = z;
            }
        }

        // Client: decrypt + output
        Plaintext res;
        cc->Decrypt(keys.secretKey, cur, &res);
        res->SetLength(m.num_classes);
        std::vector<double> logits = res->GetRealPackedValue();

        out << id;
        for (int i = 0; i < m.num_classes; ++i)
            out << "\t" << logits[i];
        out << "\n";
        out.flush();

        ++done;
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::cerr << "[" << done << "/" << total << "] sample " << id << " complete in "
                   << std::fixed << std::setprecision(1) << secs << " sec\n";
    }

    std::cerr << "Inference results were written to " << outPath << "\n";

    return 0;
}
