//
// Created by Maryan Morel on 15/05/2017.
//

#ifndef LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_
#define LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_

// License: BSD 3 clause

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "tick/base/base.h"

class LongitudinalFeaturesLagger {
 protected:
  ulong n_intervals;
  SArrayULongPtr n_lags;
  ulong n_features;
  ulong n_lagged_features;
  SArrayULongPtr col_offset;

 public:
  // This exists solely for cereal/swig
  LongitudinalFeaturesLagger() = default;

  LongitudinalFeaturesLagger(ulong n_intervals,
                             SArrayULongPtr n_lags);

  void compute_col_offset(SArrayULongPtr n_lags);

  void dense_lag_preprocessor(ArrayDouble2d &features,
                              ArrayDouble2d &out,
                              ulong censoring) const;

  void sparse_lag_preprocessor(ArrayULong &row, ArrayULong &col,
                               ArrayDouble &data, ArrayULong &out_row,
                               ArrayULong &out_col, ArrayDouble &out_data,
                               ulong censoring) const;

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_intervals));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));

    Array<ulong> temp_n_lags, temp_col_offset;
    ar(cereal::make_nvp("n_lags", temp_n_lags));

    n_lags = temp_n_lags.as_sarray_ptr();
    col_offset = temp_col_offset.as_sarray_ptr();
  }


  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(n_intervals));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));
    ar(cereal::make_nvp("n_lags", *n_lags));
    ar(cereal::make_nvp("col_offset", *col_offset));
  }
};

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_
