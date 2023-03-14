// License: BSD 3 clause

//
// Created by Maryan Morel on 15/05/2017.
//

#include "tick/preprocessing/longitudinal_features_lagger.h"


LongitudinalFeaturesLagger::LongitudinalFeaturesLagger(
    ulong n_intervals,
    SArrayULongPtr _n_lags)
    : n_intervals(n_intervals),
      n_lags(_n_lags),
      n_features(_n_lags->size()),
      n_lagged_features(_n_lags->size() + _n_lags->sum()) {
  if (n_lags != nullptr) compute_col_offset(n_lags);
}

void LongitudinalFeaturesLagger::compute_col_offset(const SArrayULongPtr n_lags) {
  ArrayULong col_offset_temp = ArrayULong(n_lags->size());
  col_offset_temp.init_to_zero();
  for (ulong i(1); i < n_lags->size(); i++) {
    if ((*n_lags)[i] >= n_intervals) {
      TICK_ERROR("n_lags elements must be between 0 and (n_intervals - 1).");
    }
    col_offset_temp[i] = col_offset_temp[i - 1] + (*n_lags)[i-1] + 1;
  }
  col_offset = col_offset_temp.as_sarray_ptr();
}

void LongitudinalFeaturesLagger::dense_lag_preprocessor(ArrayDouble2d &features,
                                                        ArrayDouble2d &out,
                                                        ulong censoring) const {
  if (n_intervals != features.n_rows()) {
    TICK_ERROR("Features matrix rows count should match n_intervals.");
  }
  if (n_features != features.n_cols()) {
    TICK_ERROR("Features matrix column count should match n_lags length.");
  }
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
      col = (*col_offset)[feature];
      // use view_row instead of (row, feature) to be const
      value = view_row(features, row)[feature];
      max_col = col + n_cols_feature;
      if (value != 0) {
        while (row < censoring && col < max_col) {
          out[row * n_lagged_features + col] = value;
          row++;
          col++;
        }
      }
    }
  }
}

void LongitudinalFeaturesLagger::sparse_lag_preprocessor(ArrayULong &row,
                                                         ArrayULong &col,
                                                         ArrayDouble &data,
                                                         ArrayULong &out_row,
                                                         ArrayULong &out_col,
                                                         ArrayDouble &out_data,
                                                         ulong censoring) const {
  // TODO: add checks here ? Or do them in Python ?
  ulong j(0), r, c, offset, new_col, max_col;
  double value;

  for (ulong i = 0; i < data.size(); i++) {
    value = data[i];
    r = row[i];
    c = col[i];
    offset = (*col_offset)[c];
    max_col = offset + (*n_lags)[c] + 1;
    new_col = offset;

    while (r < censoring && new_col < max_col) {
      out_row[j] = r;
      out_col[j] = new_col;
      out_data[j] = value;
      r++;
      new_col++;
      j++;
    }
  }
}
