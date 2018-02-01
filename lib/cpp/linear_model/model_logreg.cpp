// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "tick/linear_model/model_logreg.h"

template <class T, class K>
TModelLogReg<T, K>::TModelLogReg(
  const std::shared_ptr<BaseArray2d<K> > features,
  const std::shared_ptr<SArray<K> > labels,
  const bool fit_intercept,
  const int n_threads
) : TModelLabelsFeatures<T, K>(features, labels),
    TModelGeneralizedLinear<T, K>(features, labels, fit_intercept, n_threads)
{}

ModelLogReg::ModelLogReg(
  const SBaseArrayDouble2dPtr features,
  const SArrayDoublePtr labels,
  const bool fit_intercept,
  const int n_threads
) : TModelLabelsFeatures<double, double>(features, labels),
    TModelLogReg<double>(features, labels, fit_intercept, n_threads)
{}

const char *ModelLogReg::get_class_name() const {
  return "ModelLogReg";
}

template <class T, class K>
void
TModelLogReg<T, K>::sigmoid(const Array<K> &x, Array<K> &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = sigmoid(x[i]);
  }
}

template <class T, class K>
void
TModelLogReg<T, K>::logistic(const Array<K> &x, Array<K> &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = logistic(x[i]);
  }
}

void
ModelLogReg::sigmoid(const ArrayDouble &x, ArrayDouble &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = sigmoid(x[i]);
  }
}

void
ModelLogReg::logistic(const ArrayDouble &x, ArrayDouble &out) {
  for (ulong i = 0; i < x.size(); ++i) {
    out[i] = logistic(x[i]);
  }
}


template <class T, class K>
K
TModelLogReg<T, K>::loss_i(const ulong i, const Array<T> &coeffs) {
  double z_i = get_inner_prod(i, coeffs);
  z_i *= get_label(i);
  return logistic(z_i);
}

template <class T, class K>
K
TModelLogReg<T, K>::grad_i_factor(const ulong i, const Array<T> &coeffs) {
  // The label in { -1, 1 }
  const K y_i = get_label(i);
  // Contains x_i^T w + b
  const K z_i = get_inner_prod(i, coeffs);

  return y_i * (sigmoid(y_i * z_i) - 1);
}

template <class T, class K>
K
TModelLogReg<T, K>::sdca_dual_min_i(const ulong i,
                                    const K dual_i,
                                    const Array<K> &primal_vector,
                                    const K previous_delta_dual_i,
                                    K l_l2sq) {
  compute_features_norm_sq();
  K epsilon = 1e-1;
  K normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const K primal_dot_features = get_inner_prod(i, primal_vector);
  const K label = get_label(i);
  K new_dual_times_label{0.};

  // initial delta dual as suggested in original paper
  // http://www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf 6.2
  K delta_dual = label / (1. + exp(primal_dot_features * label)) - dual_i;
  delta_dual /= std::max(1., 0.25 + normalized_features_norm);

  for (int j = 0; j < 10; ++j) {
    K new_dual = dual_i + delta_dual;
    new_dual_times_label = new_dual * label;
    // Check we are in the correct bounds
    if (new_dual_times_label <= 0) {
      new_dual = epsilon / label;
      delta_dual = new_dual - dual_i;
      new_dual_times_label = new_dual * label;
      epsilon *= 1e-1;
    }
    if (new_dual_times_label >= 1) {
      new_dual = (1 - epsilon) / label;
      delta_dual = new_dual - dual_i;
      new_dual_times_label = new_dual * label;
      epsilon *= 1e-1;
    }

    // Do newton descent
    // Logistic loss part
    K f_prime = label * (log(new_dual_times_label) - log(1 - new_dual_times_label));
    K f_second = 1 / (new_dual_times_label * (1 - new_dual_times_label));

    // Ridge regression part
    f_prime += normalized_features_norm * delta_dual + primal_dot_features;
    f_second += normalized_features_norm;

    delta_dual -= f_prime / f_second;
    new_dual = dual_i + delta_dual;
    new_dual_times_label = new_dual * label;

    if (std::abs(f_prime / f_second) < 1e-10) {
      break;
    }
  }
  // Check we are in the correct bounds
  if (new_dual_times_label <= 0) {
    double new_dual = epsilon / label;
    delta_dual = new_dual - dual_i;
    new_dual_times_label = new_dual * label;
  }
  if (new_dual_times_label >= 1) {
    double new_dual = (1 - epsilon) / label;
    delta_dual = new_dual - dual_i;
  }

  return delta_dual;
}

template <class T, class K>
void
TModelLogReg<T, K>::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = Array<K>(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = (features_norm_sq[i] + 1) / 4;
      } else {
        lip_consts[i] = features_norm_sq[i] / 4;
      }
    }
  }
}


template class TModelLogReg<double, double>;
template class TModelLogReg<float , float>;
