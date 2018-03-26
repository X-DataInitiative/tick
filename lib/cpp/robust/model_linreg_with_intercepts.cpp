// License: BSD 3 clause

#include "tick/robust/model_linreg_with_intercepts.h"

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

template class DLL_PUBLIC TModelLinRegWithIntercepts<double>;
template class DLL_PUBLIC TModelLinRegWithIntercepts<float>;
