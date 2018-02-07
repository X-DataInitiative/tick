// License: BSD 3 clause

#include "tick/robust/model_absolute_regression.h"

template <class T>
TModelAbsoluteRegression<T>::TModelAbsoluteRegression(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads) {}

template <class T>
const char *TModelAbsoluteRegression<T>::get_class_name() const {
  return "ModelAbsoluteRegression";
}

template <class T>
T TModelAbsoluteRegression<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  return std::abs(get_inner_prod(i, coeffs) - get_label(i));
}

template <class T>
T TModelAbsoluteRegression<T>::grad_i_factor(const ulong i,
                                             const Array<T> &coeffs) {
  const T d = get_inner_prod(i, coeffs) - get_label(i);
  if (d > 0) {
    return 1;
  } else {
    if (d < 0) {
      return -1;
    } else {
      return 0;
    }
  }
}

template class TModelAbsoluteRegression<double>;
template class TModelAbsoluteRegression<float>;
