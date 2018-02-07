// License: BSD 3 clause

#include "tick/robust/model_linreg_with_intercepts.h"

template <class T>
TModelLinRegWithIntercepts<T>::TModelLinRegWithIntercepts(
    const std::shared_ptr<BaseArray2d<T> > features,
    const std::shared_ptr<SArray<T> > labels, const bool fit_intercept,
    const int n_threads)
    : TModelLabelsFeatures<T>(features, labels),
      TModelGeneralizedLinear<T>(features, labels, fit_intercept, n_threads),
      TModelGeneralizedLinearWithIntercepts<T>(features, labels, fit_intercept,
                                               n_threads),
      TModelLinReg<T>(features, labels, fit_intercept, n_threads) {}

template <class T>
const char *TModelLinRegWithIntercepts<T>::get_class_name() const {
  return "ModelLinRegWithIntercepts";
}

template <class T>
void TModelLinRegWithIntercepts<T>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<T>(get_n_samples());
    T c = 1;
    if (use_intercept()) {
      c = 2;
    }
    for (ulong i = 0; i < get_n_samples(); ++i) {
      lip_consts[i] = get_features_norm_sq()[i] + c;
    }
  }
}

template class TModelLinRegWithIntercepts<double>;
template class TModelLinRegWithIntercepts<float>;
