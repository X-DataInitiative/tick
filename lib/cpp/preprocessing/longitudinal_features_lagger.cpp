// License: BSD 3 clause

//
// Created by Maryan Morel on 15/05/2017.
//

#include "tick/preprocessing/longitudinal_features_lagger.h"

LongitudinalFeaturesLagger::LongitudinalFeaturesLagger(
<<<<<<< ours
    const SBaseArrayDouble2dPtrList1D &features, const ulong n_lags)
||||||| ancestor
    const SBaseArrayDouble2dPtrList1D &features,
    const ulong n_lags)
=======
    const SBaseArrayDouble2dPtrList1D &features,
    const SArrayULongPtr n_lags)
>>>>>>> theirs
    : n_intervals(features[0]->n_rows()),
      n_lags(n_lags),
      n_samples(features.size()),
      n_observations(n_samples * n_intervals),
      n_features(features[0]->n_cols()),
      n_lagged_features(n_lags->sum() + n_lags->size()) {
  col_offset = ArrayULong(n_lags->size());
  col_offset.init_to_zero();
  if (n_features != n_lags->size()) {
    TICK_ERROR("Features matrix column number should match n_lags length.");
  }
  if ((*n_lags)[0] >= n_intervals) {
    TICK_ERROR("n_lags elements must be between 0 and (n_intervals - 1).");
  }
  for (ulong i(1); i < n_lags->size(); i++) {
    if ((*n_lags)[i] >= n_intervals) {
      TICK_ERROR("n_lags elements must be between 0 and (n_intervals - 1).");
    }
    col_offset[i] = col_offset[i - 1] + (*n_lags)[i-1] + 1;
  }
}

void LongitudinalFeaturesLagger::dense_lag_preprocessor(ArrayDouble2d &features,
                                                        ArrayDouble2d &out,
                                                        ulong censoring) const {
  if (out.n_cols() != n_lagged_features) {
    TICK_ERROR(
        "n_columns of &out should be equal to n_features + sum(n_lags).");
  }
  if (out.n_rows() != n_intervals) {
    TICK_ERROR("n_rows of &out is inconsistent with n_intervals");
  }
  ulong n_cols_feature, row, col, max_col;
  double value;
  for (ulong feature = 0; feature < n_features; feature++) {
    n_cols_feature = (*n_lags)[feature] + 1;
    for (ulong j = 0; j < n_intervals; j++) {
      row = j;
      col = col_offset[feature];
      value = features(row, feature);
      max_col = col + n_cols_feature;
      if (value != 0) {
<<<<<<< ours
        while (row < censoring && row / n_intervals == sample &&
               col / (n_lags + 1) == feature) {
||||||| ancestor
        while (row < censoring &&
            row / n_intervals == sample &&
            col / (n_lags + 1) == feature) {
=======
        while (row < censoring && col < max_col) {
>>>>>>> theirs
          out[row * n_lagged_features + col] = value;
          row++;
          col++;
        }
      }
    }
  }
}

<<<<<<< ours
void LongitudinalFeaturesLagger::sparse_lag_preprocessor(
    ArrayULong &row, ArrayULong &col, ArrayDouble &data, ArrayULong &out_row,
    ArrayULong &out_col, ArrayDouble &out_data, ulong censoring) const {
  ulong j = 0;
  for (ulong i = 0; i < row.size(); i++) {
    double value = data[i];
    ulong r = row[i];
    ulong c = col[i] * (n_lags + 1);
    ulong sample = r / n_intervals;
    ulong feature = c / (n_lags + 1);
    while (r < censoring && r / n_intervals == sample &&
           c / (n_lags + 1) == feature) {
||||||| ancestor
void LongitudinalFeaturesLagger::sparse_lag_preprocessor(ArrayULong &row,
                                                         ArrayULong &col,
                                                         ArrayDouble &data,
                                                         ArrayULong &out_row,
                                                         ArrayULong &out_col,
                                                         ArrayDouble &out_data,
                                                         ulong censoring) const {
  ulong j = 0;
  for (ulong i = 0; i < row.size(); i++) {
    double value = data[i];
    ulong r = row[i];
    ulong c = col[i] * (n_lags + 1);
    ulong sample = r / n_intervals;
    ulong feature = c / (n_lags + 1);
    while (r < censoring &&
        r / n_intervals == sample &&
        c / (n_lags + 1) == feature) {
=======
void LongitudinalFeaturesLagger::sparse_lag_preprocessor(ArrayULong &row,
                                                         ArrayULong &col,
                                                         ArrayDouble &data,
                                                         ArrayULong &out_row,
                                                         ArrayULong &out_col,
                                                         ArrayDouble &out_data,
                                                         ulong censoring) const {
  ulong j(0), r, c, offset, new_col, max_col;
  double value;

  for (ulong i = 0; i < data.size(); i++) {
    value = data[i];
    r = row[i];
    c = col[i];
    offset = col_offset[c];
    max_col = offset + (*n_lags)[c] + 1;
    new_col = offset;

    while (r < censoring && new_col < max_col) {
>>>>>>> theirs
      out_row[j] = r;
      out_col[j] = new_col;
      out_data[j] = value;
      r++;
      new_col++;
      j++;
    }
  }
}
