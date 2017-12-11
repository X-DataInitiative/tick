// License: BSD 3 clause

#include "tick/robust/model_generalized_linear_with_intercepts.h"

ModelGeneralizedLinearWithIntercepts::ModelGeneralizedLinearWithIntercepts(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const bool fit_intercept,
    const int n_threads)
    : ModelGeneralizedLinear(features, labels, fit_intercept, n_threads) {}

const char *ModelGeneralizedLinearWithIntercepts::get_class_name() const {
  return "ModelGeneralizedLinear";
}

double ModelGeneralizedLinearWithIntercepts::get_inner_prod(const ulong i,
                                                            const ArrayDouble &coeffs) const {
  const BaseArrayDouble x_i = get_features(i);
  const ArrayDouble weights = view(coeffs, 0, n_features);
  if (fit_intercept) {
    return x_i.dot(weights) + coeffs[n_features] + coeffs[n_features + 1 + i];
  } else {
    return x_i.dot(weights) + coeffs[n_features + i];
  }
}

void ModelGeneralizedLinearWithIntercepts::compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                                                          ArrayDouble &out, const bool fill) {
  const BaseArrayDouble x_i = get_features(i);
  const double alpha_i = grad_i_factor(i, coeffs);
  ArrayDouble out_weights = view(out, 0, n_features);

  if (fit_intercept) {
    ArrayDouble out_intercepts = view(out, n_features + 1, n_samples + n_features + 1);
    if (fill) {
      out_weights.mult_fill(x_i, alpha_i);
      out_intercepts.fill(0);
      out_intercepts[i] = alpha_i;
      out[n_features] = alpha_i;
    } else {
      out_weights.mult_incr(x_i, alpha_i);
      out_intercepts[i] += alpha_i;
      out[n_features] += alpha_i;
    }
  } else {
    ArrayDouble out_intercepts = view(out, n_features, n_samples + n_features);
    if (fill) {
      out_weights.mult_fill(x_i, alpha_i);
      out_intercepts.fill(0);
      out_intercepts[i] = alpha_i;
    } else {
      out_weights.mult_incr(x_i, alpha_i);
      out_intercepts[i] += alpha_i;
    }
  }
}

void ModelGeneralizedLinearWithIntercepts::grad(const ArrayDouble &coeffs,
                                                ArrayDouble &out) {
  out.fill(0.0);
  parallel_map_array<ArrayDouble>(n_threads,
                                  n_samples,
                                  [](ArrayDouble &r, const ArrayDouble &s) {
                                    r.mult_incr(s,
                                                1.0);
                                  },
                                  &ModelGeneralizedLinearWithIntercepts::inc_grad_i,
                                  this,
                                  out,
                                  coeffs);

  const double one_over_n_samples = 1.0 / n_samples;
  out *= one_over_n_samples;
}

double ModelGeneralizedLinearWithIntercepts::loss(const ArrayDouble &coeffs) {
  return parallel_map_additive_reduce(n_threads, n_samples,
                                      &ModelGeneralizedLinearWithIntercepts::loss_i,
                                      this, coeffs)
      / n_samples;
}
