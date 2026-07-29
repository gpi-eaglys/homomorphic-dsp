// halevi_shoup.h -- shared Halevi-Shoup diagonal-packed matrix-vector product (CKKS)
//
// Extracted from exp03-mnist/src/cpp/infer_mlp.cpp so exp04-mnist's infer_cnn.cpp can
// reuse the exact same dense-FC-layer evaluation instead of duplicating it.
#pragma once
#include "openfhe.h"

using namespace lbcrypto;

// out = W . x, where `diags[i]` is the i-th diagonal of W (dim entries, i in [0, dim)):
//     W @ x  =  sum_i  diag_i(W)  *  rotate(x, i)
inline Ciphertext<DCRTPoly> matVec(
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
