//
// Created by Maryan Morel on 15/05/2017.
//

#ifndef LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_
#define LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>

class LongitudinalFeaturesLagger {
 protected:
  ulong n_intervals;
  ulong n_lags;
  ulong n_samples;
  ulong n_observations;
  ulong n_features;
  ulong n_lagged_features;

 public:
  LongitudinalFeaturesLagger(const SBaseArrayDouble2dPtrList1D &features,
                             const ulong n_lags);

  void dense_lag_preprocessor(ArrayDouble2d &features,
                              ArrayDouble2d &out,
                              ulong censoring) const;

  void sparse_lag_preprocessor(ArrayULong &row,
                               ArrayULong &col,
                               ArrayDouble &data,
                               ArrayULong &out_row,
                               ArrayULong &out_col,
                               ArrayDouble &out_data,
                               ulong censoring) const;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(n_intervals));
    ar(CEREAL_NVP(n_lags));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_observations));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));
  }
};

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_H_
