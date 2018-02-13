// License: BSD 3 clause

//
// Created by Maryan Morel on 18/05/2017.
//

#include "tick/survival/model_sccs.h"
#include <cmath>
#include "tick/base/base.h"

// Remark: in this model, i represents a patient, not an observation (i, b)

template <class T>
TModelSCCS<T>::TModelSCCS(const SBaseArrayT2dPtrList1D &features,
                          const SArrayIntPtrList1D &labels,
                          const SBaseArrayULongPtr censoring, ulong n_lags)
    : n_intervals(features[0]->n_rows()),
      n_lags(n_lags),
      n_samples(features.size()),
      n_observations(n_samples * n_intervals),
      n_lagged_features(features[0]->n_cols()),
      n_features(n_lags > 0 ? n_lagged_features / (n_lags + 1)
                            : n_lagged_features),
      labels(labels),
      features(features),
      censoring(censoring) {
  if (n_lags >= n_intervals)
    TICK_ERROR("ModelSCCS requires n_lags < n_intervals");

  if (n_samples != labels.size() || n_samples != (*censoring).size())
    TICK_ERROR("features, labels and censoring should have equal length.");

  if (n_lags > 0 && n_lagged_features % (n_lags + 1) != 0)
    TICK_ERROR(
        "n_lags should be a divisor of the number of feature matrices "
        "columns.");

  for (ulong i(0); i < n_samples; i++) {
    if (features[i]->n_rows() != n_intervals)
      TICK_ERROR("All feature matrices should have " << n_intervals << " rows");

    if (features[i]->n_cols() != n_lagged_features)
      TICK_ERROR("All feature matrices should have " << n_lagged_features
                                                     << " cols");

    if (labels[i]->size() != n_intervals)
      TICK_ERROR("All labels should have " << n_intervals << " rows");
  }
}

template <class T>
T TModelSCCS<T>::loss(const Array<T> &coeffs) {
  T loss = 0;
  for (ulong i = 0; i < n_samples; ++i) loss += loss_i(i, coeffs);

  return loss / n_samples;
}

template <class T>
T TModelSCCS<T>::loss_i(const ulong i, const Array<T> &coeffs) {
  T loss = 0;
  Array<T> inner_prod(n_intervals), softmax(n_intervals);
  ulong max_interval = get_max_interval(i);

  for (ulong t = 0; t < max_interval; t++)
    inner_prod[t] = get_inner_prod(i, t, coeffs);
  for (ulong t = max_interval; t < n_intervals; t++) inner_prod[t] = 0;

  softMax(inner_prod, softmax);

  for (ulong t = 0; t < max_interval; t++)
    loss -= get_longitudinal_label(i, t) * log(softmax[t]);

  return loss;
}

template <class T>
void TModelSCCS<T>::grad(const Array<T> &coeffs, Array<T> &out) {
  out.init_to_zero();
  Array<T> buffer(out.size());  // is init to 0 in grad_i

  for (ulong i = 0; i < n_samples; ++i) {
    grad_i(i, coeffs, buffer);
    out.mult_incr(buffer, 1);
  }

  for (ulong j = 0; j < out.size(); ++j) {
    out[j] /= n_samples;
  }
}

template <class T>
void TModelSCCS<T>::grad_i(const ulong i, const Array<T> &coeffs,
                           Array<T> &out) {
  out.init_to_zero();
  Array<T> inner_prod(n_intervals);
  Array<T> buffer(n_lagged_features);
  buffer.init_to_zero();
  ulong max_interval = get_max_interval(i);

  for (ulong t = 0; t < max_interval; t++)
    inner_prod[t] = get_inner_prod(i, t, coeffs);

  if (max_interval < n_intervals)
    view(inner_prod, max_interval, n_intervals).fill(0);  // TODO

  T x_max = inner_prod.max();
  T sum_exp = sumExpMinusMax(inner_prod, x_max);

  T multiplier = 0;  // need a T instead of long T for mult_incr
  for (ulong t = 0; t < max_interval; t++) {
    multiplier = exp(inner_prod[t] - x_max) / sum_exp;  // overflow-proof
    buffer.mult_incr(get_longitudinal_features(i, t), multiplier);
  }

  T label = 0;
  for (ulong t = 0; t < max_interval; t++) {
    label = get_longitudinal_label(i, t);
    if (label != 0) {
      out.mult_add_mult_incr(get_longitudinal_features(i, t), -label, buffer,
                             label);
    }
  }
}

template <class T>
void TModelSCCS<T>::compute_lip_consts() {
  lip_consts = Array<T>(n_samples);
  lip_consts.init_to_zero();
  BaseArray<T> row, other_row;

  T max_sq_norm, sq_norm;

  for (ulong sample = 0; sample < n_samples; sample++) {
    max_sq_norm = 0;
    ulong max_interval = get_max_interval(sample);
    for (ulong t = 0; t < max_interval; t++) {
      row = get_longitudinal_features(sample, t);
      // Lipschitz constant = 0 if Y_{sample, t} = 0
      if (get_longitudinal_label(sample, t) > 0) {
        for (ulong k = 0; k < max_interval; k++) {
          other_row = get_longitudinal_features(sample, k);
          sq_norm = 0;
          for (ulong feature = 0; feature < n_lagged_features; feature++) {
            sq_norm += pow(row.value(feature) - other_row.value(feature), 2.0);
          }
          max_sq_norm = sq_norm > max_sq_norm ? sq_norm : max_sq_norm;
        }
      }
    }
    lip_consts[sample] = max_sq_norm / 4;
  }
}

template <class T>
T TModelSCCS<T>::get_inner_prod(const ulong i, const ulong t,
                                const Array<T> &coeffs) const {
  BaseArray<T> sample = get_longitudinal_features(i, t);
  return sample.dot(coeffs);
}

template class TModelSCCS<double>;
template class TModelSCCS<float>;
