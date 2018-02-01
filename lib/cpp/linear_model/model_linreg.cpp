// License: BSD 3 clause

//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#include "tick/linear_model/model_linreg.h"

template <class T, class K>
TModelLinReg<T, K>::TModelLinReg(
  const std::shared_ptr<BaseArray2d<K> > features,
  const std::shared_ptr<SArray<K> > labels,
  const bool fit_intercept,
  const int n_threads
) : TModelLabelsFeatures<T, K>(features, labels),
    TModelGeneralizedLinear<T, K>(features, labels, fit_intercept, n_threads)
{}

ModelLinReg::ModelLinReg(
  const SBaseArrayDouble2dPtr features,
  const SArrayDoublePtr labels,
  const bool fit_intercept,
  const int n_threads
) : TModelLabelsFeatures<double, double>(features, labels),
    TModelGeneralizedLinear<double, double>(features, labels, fit_intercept, n_threads),
    TModelLinReg<double, double>(features, labels, fit_intercept, n_threads)
{}

const char *ModelLinReg::get_class_name() const {
  return "ModelLinReg";
}

template <class T, class K>
K
TModelLinReg<T, K>::sdca_dual_min_i(const ulong i,
                                    const K dual_i,
                                    const Array<K> &primal_vector,
                                    const K previous_delta_dual_i,
                                    K l_l2sq) {
  compute_features_norm_sq();
  K normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const K primal_dot_features = get_inner_prod(i, primal_vector);
  const K label = get_label(i);
  const K delta_dual = -(dual_i + primal_dot_features - label) / (1 + normalized_features_norm);
  return delta_dual;
}

template <class T, class K>
K
TModelLinReg<T, K>::loss_i(const ulong i,
                           const Array<T> &coeffs) {
  // Compute x_i^T \beta + b
  const K z = get_inner_prod(i, coeffs);
  const K d = get_label(i) - z;
  return d * d / 2;
}

template <class T, class K>
K
TModelLinReg<T, K>::grad_i_factor(const ulong i,
                                  const Array<T> &coeffs) {
  const K z = get_inner_prod(i, coeffs);
  return z - get_label(i);
}

template <class T, class K>
void
TModelLinReg<T, K>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<K>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = features_norm_sq[i] + 1;
      } else {
        lip_consts[i] = features_norm_sq[i];
      }
    }
  }
}

template class TModelLinReg<double, double>;
template class TModelLinReg<float , float>;
