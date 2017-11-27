// License: BSD 3 clause

//
// Created by Maryan Morel on 15/05/2017.
//

#include "tick/preprocessing/longitudinal_features_lagger.h"

LongitudinalFeaturesLagger::LongitudinalFeaturesLagger(
    const SBaseArrayDouble2dPtrList1D &features,
    const ulong n_lags)
    : n_intervals(features[0]->n_rows()),
      n_lags(n_lags),
      n_samples(features.size()),
      n_observations(n_samples * n_intervals),
      n_features(features[0]->n_cols()),
      n_lagged_features(n_features * (n_lags + 1)) {
  if (n_lags >= n_intervals) {
    TICK_ERROR("n_lags must be between 0 and (n_intervals - 1)");
  }
}

void LongitudinalFeaturesLagger::dense_lag_preprocessor(
    ArrayDouble2d &features,
    ArrayDouble2d &out,
    ulong censoring) const {
  if (out.n_cols() != n_lagged_features) {
    TICK_ERROR(
        "n_columns of &out is inconsistent with n_features * (n_lags + 1).");
  }
  if (out.n_rows() != n_intervals) {
    TICK_ERROR(
        "n_rows of &out is inconsistent with n_intervals");
  }
  ulong sample, row, col;
  double value;
  for (ulong feature = 0; feature < n_features; feature++) {
    for (ulong j = 0; j < n_intervals; j++) {
      row = j;
      sample = row / n_intervals;
      col = feature * (n_lags + 1);
      value = features(row, feature);
      if (value != 0) {
        while (row < censoring &&
            row / n_intervals == sample &&
            col / (n_lags + 1) == feature) {
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
      out_row[j] = r;
      out_col[j] = c;
      out_data[j] = value;
      r++;
      c++;
      j++;
    }
  }
}
