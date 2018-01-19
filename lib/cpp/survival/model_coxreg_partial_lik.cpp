// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 12/04/2016.
//

#include "tick/survival/model_coxreg_partial_lik.h"

#include <cmath>

// Include this to get DBL_MAX, DBL_MIN and DBL_MIN_EXP
#include <cfloat>

ModelCoxRegPartialLik::ModelCoxRegPartialLik(const SBaseArrayDouble2dPtr features,
                                             const SArrayDoublePtr times_,
                                             const SArrayUShortPtr censoring_) {
    n_samples = features->n_rows();
    n_features = features->n_cols();
    n_failures = 0;

    // Make copies for times_ and censoring_ (we keep sorted versions of them, but not for
    // the features, as it might get very large)
    this->features = features;
    times = ArrayDouble(*times_);
    censoring = ArrayUShort(n_samples);

    // Will contain inner products for loss and gradient computations
    inner_prods = ArrayDouble(n_samples);
    // Used for gradient computations
    s1 = ArrayDouble(n_features);

    // Get the indices that sort the times by decreasing order in idx
    idx = ArrayULong(n_samples);
    times.sort(idx, false);

    // Count number of true failure times, and sort censoring by with respect to decreasing times
    for (ulong i=0; i < n_samples; i++) {
        censoring[i] = (*censoring_)[idx[i]];
        ushort is_failure = censoring[i];
        if (is_failure != 0) {
            n_failures++;
        }
    }

    // If the last time is a true failure time, then the likelihood for it
    // is constant and equal to zero, which leads to errors in the
    // gradients computations, hence we change to false in this case
    if (censoring[0] == 1) {
        censoring[0] = 0;
        n_failures--;
    }

    // Instantiate and fill the indices of true sorted times (sorted w.r.t time)
    idx_failures = ArrayULong(static_cast<ulong>(n_failures));

    ulong i_failure = 0;
    for (ulong i=0; i < n_samples; i++) {
        ushort is_failure = get_censoring(i);
        if (is_failure != 0) {
            idx_failures[i_failure] = i;
            i_failure++;
        }
    }
}


double ModelCoxRegPartialLik::loss(const ArrayDouble &coeffs) {
    const ulong n_failures_minus_one = n_failures - 1;
    // Compute all the inner products and maintain the maximal one
    double max_inner_prod = -DBL_MAX;
    for (ulong i = 0; i < n_samples; ++i) {
        const double inner_prod = get_feature(i).dot(coeffs);
        inner_prods[i] = inner_prod;
        if (inner_prod > max_inner_prod) {
            max_inner_prod = inner_prod;
        }
    }
    double log_lik = 0;
    double s = DBL_MIN;
    const ulong idx0 = get_idx_failure(0);
    for (ulong i = 0; i <= idx0; ++i) {
        const double diff = inner_prods[i] - max_inner_prod;
        s += exp(diff);
    }
    for (ulong k = 0; k < n_failures; ++k) {
        const ulong idx = get_idx_failure(k);
        log_lik += log(s) - inner_prods[idx] + max_inner_prod;
        if (k != n_failures_minus_one) {
            const ulong idx_next = get_idx_failure(k + 1) + 1;
            for (ulong i = idx + 1; i < idx_next; ++i) {
                const double diff = inner_prods[i] - max_inner_prod;
                if (diff > DBL_MIN_EXP) {
                    s += exp(diff);
                }
            }
        }
    }
    return log_lik / n_failures;
}

void ModelCoxRegPartialLik::grad(const ArrayDouble &coeffs, ArrayDouble &out) {
    const ulong n_failures_minus_one = n_failures - 1;
    // grad must be filled with 0
    out.init_to_zero();
    s1.init_to_zero();
    inner_prods.init_to_zero();

    // Initialize s to a very small positive number (to avoid division by
    // 0 in weird cases)
    double s2 = DBL_MIN;

    // Compute first all inner products and keep the maximum
    // (to avoid overflow)
    double max_inner_prod = -DBL_MAX;
    for (ulong i = 0; i < n_samples; ++i) {
        const double inner_prod = get_feature(i).dot(coeffs);
        inner_prods[i] = inner_prod;
        if (inner_prod > max_inner_prod) {
            max_inner_prod = inner_prod;
        }
    }

    const ulong idx0 = get_idx_failure(0);
    for (ulong i = 0; i <= idx0; ++i) {
        const double diff = inner_prods[i] - max_inner_prod;
        const double exp_diff = exp(diff);
        if (diff > DBL_MIN_EXP) {
            const BaseArrayDouble x_i = get_feature(i);
            s1.mult_incr(x_i, exp_diff);
            s2 += exp_diff;
        }
    }

    for (ulong k = 0; k < n_failures; ++k) {
        const ulong idx = get_idx_failure(k);
        const BaseArrayDouble x_idx = get_feature(idx);

        out.mult_add_mult_incr(s1, 1 / s2, x_idx, -1);
        if (k != n_failures_minus_one) {
            const ulong idx_next = get_idx_failure(k+1) + 1;
            for (ulong i = idx + 1; i < idx_next; ++i) {
                const BaseArrayDouble x_i = get_feature(i);
                double diff = inner_prods[i] - max_inner_prod;
                if (diff > DBL_MIN_EXP) {
                    const double exp_diff = exp(diff);
                    s1.mult_incr(x_i, exp_diff);
                    s2 += exp_diff;
                }
            }
        }
    }
    for (ulong j = 0; j < n_features; ++j) {
        out[j] /= n_failures;
    }
}
