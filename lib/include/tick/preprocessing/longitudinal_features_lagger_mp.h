//
// Created by Maryan Morel on 15/05/2017.
//

#ifndef LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_MP_H_
#define LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_MP_H_

// License: BSD 3 clause

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "tick/base/base.h"
#include "tick/base/serialization.h"
#include "tick/array/coo_matrix.h"
#include "tick/preprocessing/longitudinal_preprocessor.h"

class DLL_PUBLIC LongitudinalFeaturesLagger_MP : LongitudinalPreprocessor {
 protected:
  ulong n_intervals;
  SArrayULongPtr n_lags;
  ulong n_features;
  ulong n_lagged_features;
  SArrayULongPtr col_offset;
  ulong n_output_features;

 private:
  void transform_thread_sparse(std::vector<SSparseArrayDouble2dPtr> splited_features,
                               std::vector<SSparseArrayDouble2dPtr> &output,
                               std::mutex &thread_mutex, std::vector<ulong> splited_censoring);
  void transform_thread_dense(std::vector<ArrayDouble2d> splited_features,
                              std::vector<ArrayDouble2d> &output, std::mutex &thread_mutex,
                              std::vector<ulong> splited_censoring);
  ulong get_n_output_features();

 public:
  // This exists solely for cereal/swig
  LongitudinalFeaturesLagger_MP() = default;

  LongitudinalFeaturesLagger_MP(ulong n_intervals, SArrayULongPtr n_lags, size_t n_jobs = 1);

  void compute_col_offset(SArrayULongPtr n_lags);

  void dense_lag_preprocessor(ArrayDouble2d &features, ArrayDouble2d &out, ulong censoring) const;

  void sparse_lag_preprocessor(ArrayULong &row, ArrayULong &col, ArrayDouble &data,
                               ArrayULong &out_row, ArrayULong &out_col, ArrayDouble &out_data,
                               ulong censoring) const;

  SSparseArrayDouble2dPtr sparse_lagger(SSparseArrayDouble2dPtr &feature_matrix, ulong censoring_i);

  std::vector<SSparseArrayDouble2dPtr> transform(std::vector<SSparseArrayDouble2dPtr> features,
                                                     std::vector<ulong> censoring = {});
  std::vector<ArrayDouble2d> transform(std::vector<ArrayDouble2d> features,
                                       std::vector<ulong> censoring = {});

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_intervals));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(n_lagged_features));
    ar(CEREAL_NVP(n_output_features));

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
    ar(CEREAL_NVP(n_output_features));
    ar(cereal::make_nvp("n_lags", *n_lags));
    ar(cereal::make_nvp("col_offset", *col_offset));
  }
};

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_LONGITUDINAL_FEATURES_LAGGER_MP_H_
