// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

ModelGeneralizedLinear::ModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                                               const SArrayDoublePtr labels,
                                               const bool fit_intercept,
                                               const int n_threads)
  : ModelLabelsFeatures(features, labels),
    n_threads(n_threads >= 1 ? n_threads : std::thread::hardware_concurrency()),
    fit_intercept(fit_intercept),
    ready_features_norm_sq(false) {}

void ModelGeneralizedLinear::compute_features_norm_sq() {
  if (!ready_features_norm_sq) {
    features_norm_sq = ArrayDouble(n_samples);
    // TODO: How to do it in parallel ? (I'm not sure of how to do it)
    for (ulong i = 0; i < n_samples; ++i) {
      features_norm_sq[i] = view_row(*features, i).norm_sq();
    }
    ready_features_norm_sq = true;
  }
}

const char *ModelGeneralizedLinear::get_class_name() const {
  return "ModelGeneralizedLinear";
}


double ModelGeneralizedLinear::grad_i_factor(const ulong i,
                                             const ArrayDouble &coeffs) {
  std::stringstream ss;
  ss << get_class_name() << " does not implement " << __func__;
  throw std::runtime_error(ss.str());
}

void ModelGeneralizedLinear::compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                                            ArrayDouble &out, const bool fill) {
  const BaseArrayDouble x_i = get_features(i);
  const double alpha_i = grad_i_factor(i, coeffs);
  if (fit_intercept) {
    ArrayDouble out_no_interc = view(out, 0, n_features);

    if (fill) {
      out_no_interc.mult_fill(x_i, alpha_i);
      out[n_features] = alpha_i;
    } else {
      out_no_interc.mult_incr(x_i, alpha_i);
      out[n_features] += alpha_i;
    }
  } else {
    if (fill)
      out.mult_fill(x_i, alpha_i);
    else
      out.mult_incr(x_i, alpha_i);
  }
}

void ModelGeneralizedLinear::grad_i(const ulong i, const ArrayDouble &coeffs,
                                    ArrayDouble &out) {
  compute_grad_i(i, coeffs, out, true);
}

void ModelGeneralizedLinear::inc_grad_i(const ulong i, ArrayDouble &out,
                                        const ArrayDouble &coeffs) {
  compute_grad_i(i, coeffs, out, false);
}

void ModelGeneralizedLinear::grad(const ArrayDouble &coeffs,
                                  ArrayDouble &out) {
  out.fill(0.0);

  parallel_map_array<ArrayDouble>(n_threads,
                                  n_samples,
                                  [](ArrayDouble &r, const ArrayDouble &s) { r.mult_incr(s, 1.0); },
                                  &ModelGeneralizedLinear::inc_grad_i,
                                  this,
                                  out,
                                  coeffs);

  double one_over_n_samples = 1.0 / n_samples;

  out *= one_over_n_samples;
}

double ModelGeneralizedLinear::loss(const ArrayDouble &coeffs) {
  return parallel_map_additive_reduce(n_threads, n_samples, &ModelGeneralizedLinear::loss_i,
                                      this, coeffs)
    / n_samples;
}

void ModelGeneralizedLinear::sdca_primal_dual_relation(const double l_l2sq,
                                                       const ArrayDouble &dual_vector,
                                                       ArrayDouble &out_primal_vector) {
  if (dual_vector.size() != get_n_samples()) {
    TICK_ERROR("dual vector should have shape of (" << get_n_samples() << ", )");
  }
  if (out_primal_vector.size() != get_n_coeffs()) {
    TICK_ERROR("primal vector should have shape of (" << get_n_coeffs() << ", )");
  }

  const double _1_over_lbda_n = 1 / (l_l2sq * get_n_samples());
  out_primal_vector.init_to_zero();

  for (ulong i = 0; i < get_n_samples(); ++i) {
    const BaseArrayDouble feature_i = get_features(i);
    const double dual_i = dual_vector[i];
    const double factor = dual_i * _1_over_lbda_n;

    if (fit_intercept) {
      // The last coefficient of out_primal_vector is the intercept
      ArrayDouble w = view(out_primal_vector, 0, get_n_coeffs() - 1);
      w.mult_incr(feature_i, factor);
      out_primal_vector[get_n_coeffs() - 1] += factor;
    } else {
      out_primal_vector.mult_incr(feature_i, factor);
    }
  }
}

double ModelGeneralizedLinear::get_inner_prod(const ulong i, const ArrayDouble &coeffs) const {
  const BaseArrayDouble x_i = get_features(i);
  if (fit_intercept) {
    // The last coefficient of coeffs is the intercept
    const ulong size = coeffs.size();
    const ArrayDouble w = view(coeffs, 0, size - 1);
    return x_i.dot(w) + coeffs[size - 1];
  } else {
    return x_i.dot(coeffs);
  }
}
