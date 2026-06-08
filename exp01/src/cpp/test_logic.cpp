// test_logic.cpp -- verify json_mini.h parsing and the diagonal-matvec /
// square-activation math in PLAINTEXT, matching numpy. No OpenFHE needed.
#include "server/json_mini.h"
#include <iostream>
#include <vector>
#include <cmath>

// plaintext analogue of EvalRotate (cyclic left rotation by k)
std::vector<double> rot(const std::vector<double>& v, int k) {
    int n = v.size();
    std::vector<double> o(n);
    for (int i = 0; i < n; ++i) o[i] = v[(i + k) % n];
    return o;
}

// plaintext diagonal matvec, mirrors matVecDiag in fhe_server.cpp
std::vector<double> matVecDiag(const std::vector<std::vector<double>>& diags,
                               const std::vector<double>& x, int dim) {
    std::vector<double> acc(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        auto xr = rot(x, i);
        for (int j = 0; j < dim; ++j) acc[j] += diags[i][j] * xr[j];
    }
    return acc;
}

int main() {
    Model m = loadModel("model/model.json");
    std::cout << "parsed: classes=" << m.classes.size()
              << " packed_dim=" << m.packed_dim
              << " W1_diag=" << m.W1_diag.size() << "x" << m.W1_diag[0].size()
              << "\n";
    const int dim = m.packed_dim;

    // feed a fixed test vector (already "normalized" = just ramp for the test)
    std::vector<double> x(dim, 0.0);
    for (int i = 0; i < m.input_dim; ++i) x[i] = 0.1 * (i + 1);

    auto z1 = matVecDiag(m.W1_diag, x, dim);
    for (int i = 0; i < dim; ++i) z1[i] += m.b1[i];
    std::vector<double> h(dim);
    for (int i = 0; i < dim; ++i) h[i] = z1[i] * z1[i];
    auto z2 = matVecDiag(m.W2_diag, h, dim);
    for (int i = 0; i < dim; ++i) z2[i] += m.b2[i];

    std::cout << "logits:";
    for (int i = 0; i < m.num_classes; ++i) std::cout << " " << z2[i];
    std::cout << "\n";
    // write x and the expected logits for the numpy cross-check
    FILE* f = fopen("model/features.txt", "w");
    for (int i = 0; i < m.input_dim; ++i) fprintf(f, "%.10f ", x[i] * m.stdv[i] + m.mean[i]);
    fclose(f);
    return 0;
}
