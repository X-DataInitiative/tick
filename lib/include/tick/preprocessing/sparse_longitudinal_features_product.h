//
// Created by Maryan Morel on 11/05/2017.
//

#ifndef LIB_INCLUDE_TICK_PREPROCESSING_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H_
#define LIB_INCLUDE_TICK_PREPROCESSING_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H_

// License: BSD 3 clause

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "tick/base/base.h"

class SparseLongitudinalFeaturesProduct {
 protected:
  ulong n_features;

 public:
  // This exists soley for cereal/swig
  SparseLongitudinalFeaturesProduct() = default;

  explicit SparseLongitudinalFeaturesProduct(
      const ulong n_features): n_features(n_features) {}

  inline ulong get_feature_product_col(ulong col1, ulong col2,
                                       ulong n_cols) const;

  void sparse_features_product(ArrayULong &row, ArrayULong &col,
                               ArrayDouble &data, ArrayULong &out_row,
                               ArrayULong &out_col,
                               ArrayDouble &out_data) const;

  template <class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_features));
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(n_features));
  }
};

#endif  // LIB_INCLUDE_TICK_PREPROCESSING_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H_
