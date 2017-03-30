//
// Created by Stéphane GAIFFAS on 06/12/2015.
//

#include "model_generalized_linear_with_intercepts.h"

ModelGeneralizedLinearWithIntercepts::ModelGeneralizedLinearWithIntercepts(
    const SBaseArrayDouble2dPtr features,
    const SArrayDoublePtr labels,
    const int n_threads)
    : ModelGeneralizedLinear(features, labels, true, n_threads) {}

const char *ModelGeneralizedLinearWithIntercepts::get_class_name() const {
  return "ModelGeneralizedLinear";
}

double ModelGeneralizedLinearWithIntercepts::get_inner_prod(const ulong i,
                                                            const ArrayDouble &coeffs) const {
  const BaseArrayDouble x_i = get_features(i);
  const ArrayDouble coeffs_no_interc = view(coeffs, 0, n_features);
  const ArrayDouble coeffs_interc = view(coeffs, n_features, n_samples + n_features);
  return x_i.dot(coeffs_no_interc) + coeffs_interc[i];
}

void ModelGeneralizedLinearWithIntercepts::compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                                                          ArrayDouble &out, const bool fill) {
  const BaseArrayDouble x_i = get_features(i);
  const double alpha_i = grad_i_factor(i, coeffs);
  ArrayDouble out_no_interc = view(out, 0, n_features);
  ArrayDouble out_interc = view(out, n_features, n_samples + n_features);

  if (fill) {
    out_no_interc.mult_fill(x_i, alpha_i);
    out_interc.fill(0);
    out_interc[i] = alpha_i;
  } else {
    out_no_interc.mult_incr(x_i, alpha_i);
    out_interc[i] += alpha_i;
  }
}

void ModelGeneralizedLinearWithIntercepts::grad(const ArrayDouble &coeffs,
                                                ArrayDouble &out) {
  out.fill(0.0);
  parallel_map_array<ArrayDouble>(n_threads,
                                  n_samples,
                                  [](ArrayDouble &r, const ArrayDouble &s) { r.mult_incr(s, 1.0); },
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
