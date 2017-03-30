//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#include "prox_sorted_l1.h"

#include <stdio.h>
#include <cmath>

ProxSortedL1::ProxSortedL1(double strength,
                           double fdr,
                           WeightsType weights_type,
                           bool positive)
    : Prox(strength) {
    this->fdr = fdr;
    this->weights_type = weights_type;
    this->positive = positive;
    weights_ready = false;
}


ProxSortedL1::ProxSortedL1(double strength,
                           double fdr,
                           WeightsType weights_type,
                           ulong start,
                           ulong end,
                           bool positive)
    : Prox(strength, start, end) {
    this->fdr = fdr;
    this->weights_type = weights_type;
    this->positive = positive;
    weights_ready = false;
}

const std::string ProxSortedL1::get_class_name() const {
    return "ProxSortedL1";
}

void ProxSortedL1::compute_weights(void) {
    if (!weights_ready) {
        ulong size = end - start;
        weights = ArrayDouble(size);
        switch (weights_type) {
            // Benjamini - Hochberg case
            case WeightsType::bh:
                for (ulong i = 0; i < size; i++) {
                    double tmp = fdr / (2 * size);
                    weights[i] = strength * standard_normal_inv_cdf(1 - tmp * (i + 1));
                }
                break;
                // Oscar penalization case
            case WeightsType::oscar:
              std::stringstream ss;
              ss << "weights type 'oscar' is not implemented yet";
              throw std::runtime_error(ss.str());
        }
        weights_ready = true;
    }
}

void ProxSortedL1::_call(ArrayDouble &coeffs,
                         double t,
                         ArrayDouble &out,
                         ulong start,
                         ulong end) {
    // If necessary, compute weights
    compute_weights();
    ulong size = end - start;
    double thresh = t;

    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayDouble sub_out = view(out, start, end);
    sub_out.fill(0);

    ArrayDouble weights_copy(weights);

    ArrayShort sub_coeffs_sign(size);
    ArrayDouble sub_coeffs_abs(size);
    // Will contain the indexes that sort abs(sub_coeffs) in decreasing order
    ArrayULong idx(size);
    // Sort sub_coeffs with decreasing absolute values, keeping the sorting index
    ArrayDouble sub_coeffs_sorted = sort_abs(sub_coeffs, idx, false);

    ArrayDouble sub_coeffs_abs_sort(size);
    // Multiply weights by the threshold, compute abs and sign of sub_coeffs,
    // and compute abs(sub_coeffs_sorted)
    for (ulong i = 0; i < size; i++) {
        weights_copy[i] *= thresh;
        sub_coeffs_abs_sort[i] = std::abs(sub_coeffs_sorted[i]);
        double sub_coeffs_i = sub_coeffs[i];
        if (sub_coeffs_i >= 0) {
            sub_coeffs_sign[i] = 1;
            sub_coeffs_abs[i] = sub_coeffs_i;
        } else {
            sub_coeffs_sign[i] = -1;
            sub_coeffs_abs[i] = -sub_coeffs_i;
        }
    }
    // Where do the crossing occurs?
    ulong crossing = 0;
    for (ulong i = 0; i < size; i++) {
        if (sub_coeffs_abs_sort[i] > weights_copy[i]) {
            crossing = i;
        }
    }
    ulong n_sub_coeffs;
    if (crossing > 0) {
        n_sub_coeffs = crossing + 1;
    } else {
        n_sub_coeffs = size;
    }

    ArrayDouble subsub_coeffs = view(sub_coeffs_abs_sort, 0, n_sub_coeffs);
    ArrayDouble subsub_out(n_sub_coeffs);
    subsub_out.fill(0);

    prox_sorted_l1(subsub_coeffs, weights_copy, subsub_out);

    for (ulong i = 0; i < n_sub_coeffs; i++) {
        sub_out[idx[i]] = subsub_out[i];
    }
    for (ulong i = 0; i < size; i++) {
        sub_out[i] *= sub_coeffs_sign[i];
    }
}

// This piece comes from E. Candes and co-authors for SLOPE Matlab's code
// TODO: (don't forget to put proper references in the doc)
void ProxSortedL1::prox_sorted_l1(ArrayDouble &y,  // Input vector
                                  ArrayDouble &lambda,  // Thresholding vector
                                  ArrayDouble &x) const {  // output vector
    const ulong n = y.size();
    double d;
    ulong i, j, k;

    ArrayDouble s(n);
    ArrayDouble w(n);
    ArrayULong idx_i(n);
    ArrayULong idx_j(n);

    k = 0;
    for (i = 0; i < n; i++) {
        idx_i[k] = i;
        idx_j[k] = i;
        s[k] = y[i] - lambda[i];
        w[k] = s[k];
        while ((k > 0) && (w[k - 1] <= w[k])) {
            k--;
            idx_j[k] = i;
            s[k] += s[k + 1];
            w[k] = s[k] / (i - idx_i[k] + 1);
        }
        k++;
    }
    for (j = 0; j < k; j++) {
        d = w[j];
        if (d < 0) d = 0;
        for (i = idx_i[j]; i <= idx_j[j]; i++) {
            x[i] = d;
        }
    }
}


double ProxSortedL1::_value(ArrayDouble &coeffs,
                            ulong start,
                            ulong end) {
    // If necessary, compute weights
    compute_weights();
    ulong size = end - start;
    ArrayDouble sub_coeffs = view(coeffs, start, end);
    ArrayULong idx(size);
    // Sort sub_coeffs with decreasing absolute values, and keeping sorting indexes in idx
    ArrayDouble sub_coeffs_sorted = sort_abs(sub_coeffs, idx, false);
    double val = 0;
    for (ulong i = 0; i < size; i++) {
        val += weights[i] * std::abs(sub_coeffs_sorted[i]);
    }
    return val;
}
