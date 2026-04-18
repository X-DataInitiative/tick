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
  ArrayULong col_offset;
  ulong n_samples;
  ulong n_observations;
  ulong n_features;
  ulong n_lagged_features;

 public:
  LongitudinalFeaturesLagger(const SBaseArrayDouble2dPtrList1D &features,
                             const SArrayULongPtr n_lags);

  void dense_lag_preprocessor(ArrayDouble2d &features, ArrayDouble2d &out,
                              ulong censoring) const;

  void sparse_lag_preprocessor(ArrayULong &row, ArrayULong &col,
                               ArrayDouble &data, ArrayULong &out_row,
                               ArrayULong &out_col, ArrayDouble &out_data,
                               ulong censoring) const;

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_intervals));
    ArrayULong serialized_n_lags;
    ar(cereal::make_nvp("n_lags", serialized_n_lags));
    n_lags = SArrayULong::new_ptr(serialized_n_lags);
    ar(CEREAL_NVP(col_offset));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_observations));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(n_intervals));
    const ArrayULong serialized_n_lags(*n_lags);
    ar(cereal::make_nvp("n_lags", serialized_n_lags));
    ar(CEREAL_NVP(col_offset));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_observations));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(LongitudinalFeaturesLagger,
                                   cereal::specialization::member_load_save)

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_
