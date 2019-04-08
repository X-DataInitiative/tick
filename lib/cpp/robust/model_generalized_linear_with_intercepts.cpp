// License: BSD 3 clause

#include "tick/robust/model_generalized_linear_with_intercepts.h"

template <class T, class K>
T TModelGeneralizedLinearWithIntercepts<T, K>::get_inner_prod(
    const ulong i, const Array<K> &coeffs) const {
  const BaseArray<T> x_i = get_features(i);
  const Array<K> weights = view(coeffs, 0, n_features);
  if (fit_intercept) {
    return x_i.dot(weights) + coeffs[n_features] + coeffs[n_features + 1 + i];
  } else {
    return x_i.dot(weights) + coeffs[n_features + i];
  }
}

template <class T, class K>
void TModelGeneralizedLinearWithIntercepts<T, K>::compute_grad_i(
    const ulong i, const Array<K> &coeffs, Array<T> &out, const bool fill) {
  const BaseArray<T> x_i = get_features(i);
  const T alpha_i = grad_i_factor(i, coeffs);
  Array<T> out_weights = view(out, 0, n_features);

  if (fit_intercept) {
    Array<T> out_intercepts =
        view(out, n_features + 1, n_samples + n_features + 1);
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
    Array<T> out_intercepts = view(out, n_features, n_samples + n_features);
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

template <class T, class K>
void TModelGeneralizedLinearWithIntercepts<T, K>::grad(const Array<K> &coeffs,
                                                       Array<T> &out) {
  out.fill(0.0);
  parallel_map_array<Array<T>>(
      n_threads, n_samples,
      [](Array<T> &r, const Array<T> &s) { r.mult_incr(s, 1.0); },
      &TModelGeneralizedLinearWithIntercepts<T, K>::inc_grad_i, this, out,
      coeffs);

  const T one_over_n_samples = 1.0 / n_samples;
  out *= one_over_n_samples;
}

template <class T, class K>
T TModelGeneralizedLinearWithIntercepts<T, K>::loss(const Array<K> &coeffs) {
  return parallel_map_additive_reduce(
             n_threads, n_samples,
             &TModelGeneralizedLinearWithIntercepts<T, K>::loss_i, this,
             coeffs) /
         n_samples;
}

template class DLL_PUBLIC TModelGeneralizedLinearWithIntercepts<double>;
template class DLL_PUBLIC TModelGeneralizedLinearWithIntercepts<float>;

template class DLL_PUBLIC
    TModelGeneralizedLinearWithIntercepts<double, std::atomic<double>>;
template class DLL_PUBLIC
    TModelGeneralizedLinearWithIntercepts<float, std::atomic<float>>;
