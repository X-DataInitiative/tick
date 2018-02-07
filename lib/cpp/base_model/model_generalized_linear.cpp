// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

template <class T>
TModelGeneralizedLinear<T>::TModelGeneralizedLinear(
    const std::shared_ptr<BaseArray2d<T>> features,
    const std::shared_ptr<SArray<T>> labels, const bool fit_intercept,
    const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      fit_intercept(fit_intercept),
      ready_features_norm_sq(false),
      n_threads(n_threads >= 1 ? n_threads
                               : std::thread::hardware_concurrency()) {}

template <class T>
void TModelGeneralizedLinear<T>::compute_features_norm_sq() {
  if (!ready_features_norm_sq) {
    features_norm_sq = Array<T>(n_samples);
    // TODO: How to do it in parallel ? (I'm not sure of how to do it)
    for (ulong i = 0; i < n_samples; ++i) {
      features_norm_sq[i] = view_row(*features, i).norm_sq();
    }
    ready_features_norm_sq = true;
  }
}

template <class T>
T TModelGeneralizedLinear<T>::grad_i_factor(const ulong i,
                                            const Array<T> &coeffs) {
  std::stringstream ss;
  ss << get_class_name() << " does not implement " << __func__;
  throw std::runtime_error(ss.str());
}

template <class T>
void TModelGeneralizedLinear<T>::compute_grad_i(const ulong i,
                                                const Array<T> &coeffs,
                                                Array<T> &out,
                                                const bool fill) {
  const BaseArray<T> x_i = get_features(i);
  const double alpha_i = grad_i_factor(i, coeffs);
  if (fit_intercept) {
    Array<T> out_no_interc = view(out, 0, n_features);

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

template <class T>
void TModelGeneralizedLinear<T>::grad_i(const ulong i, const Array<T> &coeffs,
                                        Array<T> &out) {
  compute_grad_i(i, coeffs, out, true);
}

template <class T>
void TModelGeneralizedLinear<T>::inc_grad_i(const ulong i, Array<T> &out,
                                            const Array<T> &coeffs) {
  compute_grad_i(i, coeffs, out, false);
}

template <class T>
void TModelGeneralizedLinear<T>::grad(const Array<T> &coeffs, Array<T> &out) {
  out.fill(0.0);

  parallel_map_array<Array<T>>(
      n_threads, n_samples,
      [](Array<T> &r, const Array<T> &s) { r.mult_incr(s, 1.0); },
      &TModelGeneralizedLinear<T>::inc_grad_i, this, out, coeffs);

  double one_over_n_samples = 1.0 / n_samples;

  out *= one_over_n_samples;
}

template <class T>
T TModelGeneralizedLinear<T>::loss(const Array<T> &coeffs) {
  return parallel_map_additive_reduce(n_threads, n_samples,
                                      &TModelGeneralizedLinear<T>::loss_i, this,
                                      coeffs) /
         n_samples;
}

template <class T>
void TModelGeneralizedLinear<T>::sdca_primal_dual_relation(
    const T l_l2sq, const Array<T> &dual_vector, Array<T> &out_primal_vector) {
  if (dual_vector.size() != get_n_samples()) {
    TICK_ERROR("dual vector should have shape of (" << get_n_samples()
                                                    << ", )");
  }
  if (out_primal_vector.size() != get_n_coeffs()) {
    TICK_ERROR("primal vector should have shape of (" << get_n_coeffs()
                                                      << ", )");
  }

  const double _1_over_lbda_n = 1 / (l_l2sq * get_n_samples());
  out_primal_vector.init_to_zero();

  for (ulong i = 0; i < get_n_samples(); ++i) {
    const BaseArray<T> feature_i = get_features(i);
    const double dual_i = dual_vector[i];
    const double factor = dual_i * _1_over_lbda_n;

    if (fit_intercept) {
      // The last coefficient of out_primal_vector is the intercept
      Array<T> w = view(out_primal_vector, 0, get_n_coeffs() - 1);
      w.mult_incr(feature_i, factor);
      out_primal_vector[get_n_coeffs() - 1] += factor;
    } else {
      out_primal_vector.mult_incr(feature_i, factor);
    }
  }
}

template <class T>
T TModelGeneralizedLinear<T>::get_inner_prod(const ulong i,
                                             const Array<T> &coeffs) const {
  const BaseArray<T> x_i = get_features(i);
  if (fit_intercept) {
    // The last coefficient of coeffs is the intercept
    const ulong size = coeffs.size();
    const Array<T> w = view(coeffs, 0, size - 1);
    return x_i.dot(w) + coeffs[size - 1];
  } else {
    return x_i.dot(coeffs);
  }
}

template class TModelGeneralizedLinear<double>;
template class TModelGeneralizedLinear<float>;
