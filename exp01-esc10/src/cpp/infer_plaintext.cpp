// infer_plaintext.cpp -- Plaintext MLP inference for cross-checking the CKKS pipeline.
//
// Loads a model exported by export_ckks.py (ckks_model.json) and runs a full
// 3-layer forward pass entirely in floating-point, with no homomorphic encryption.
// Useful for verifying that (a) the JSON model loads correctly, (b) the diagonal
// matrix-vector product and square activation produce the same results as the
// Python training code, and (c) the predicted class is sane before committing to
// a slow FHE run.
//
// Usage:
//   infer_plaintext <model.json> <features.txt> [sample_key]
//
//   model.json    exported by export_ckks.py
//   features.txt  exported by export_features.py  (one sample per line: key f0 f1 ...)
//   sample_key    optional — if omitted, runs all samples and prints accuracy
#include "model.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

static std::vector<double> rot(const std::vector<double>& v, int k) {
    int n = v.size();
    std::vector<double> o(n);
    for (int i = 0; i < n; ++i) o[i] = v[(i + k) % n];
    return o;
}

static std::vector<double> matVecDiag(const std::vector<std::vector<double>>& diags,
                                      const std::vector<double>& x, int dim) {
    std::vector<double> acc(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        auto xr = rot(x, i);
        for (int j = 0; j < dim; ++j) acc[j] += diags[i][j] * xr[j];
    }
    return acc;
}

static std::vector<double> forward(const Model& m, std::vector<double> x) {
    const int dim = m.packed_dim;
    for (int i = 0; i < m.input_dim; ++i)
        x[i] = (x[i] - m.mean[i]) / m.stdv[i];

    auto z1 = matVecDiag(m.W1_diag, x, dim);
    for (int i = 0; i < dim; ++i) z1[i] += m.b1[i];
    std::vector<double> h1(dim);
    for (int i = 0; i < dim; ++i) h1[i] = z1[i] * z1[i];

    auto z2 = matVecDiag(m.W2_diag, h1, dim);
    for (int i = 0; i < dim; ++i) z2[i] += m.b2[i];
    std::vector<double> h2(dim);
    for (int i = 0; i < dim; ++i) h2[i] = z2[i] * z2[i];

    auto z3 = matVecDiag(m.W3_diag, h2, dim);
    for (int i = 0; i < dim; ++i) z3[i] += m.b3[i];
    return z3;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: infer_plaintext <model.json> <features.txt> [sample_key]\n"
                  << "  model.json    exported by export_ckks.py\n"
                  << "  features.txt  exported by export_features.py\n"
                  << "  sample_key    optional; omit to run all samples\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string featPath  = argv[2];
    std::string filterKey = (argc == 4) ? argv[3] : "";
    bool singleMode       = !filterKey.empty();

    std::cout << std::setprecision(10);
    Model m = loadModel(modelPath);
    const int dim = m.packed_dim;

    std::ifstream f(featPath);
    if (!f) { std::cerr << "cannot open features: " << featPath << "\n"; return 1; }

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (singleMode && key != filterKey) continue;

        std::vector<double> x(dim, 0.0);
        for (int i = 0; i < m.input_dim; ++i) ss >> x[i];

        auto logits = forward(m, x);

        std::cout << key;
        for (int i = 0; i < m.num_classes; ++i)
            std::cout << "  " << logits[i];
        std::cout << "\n";

        if (singleMode) return 0;
    }

    if (singleMode) {
        std::cerr << "sample key not found: " << filterKey << "\n";
        return 1;
    }

    return 0;
}
