//
// Created by Maryan Morel on 11/05/2017.
//

#ifndef TICK_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H
#define TICK_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H

// License: BSD 3 clause

#include "base.h"
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/base_class.hpp>

class SparseLongitudinalFeaturesProduct {
  protected:
    ulong n_features;

  public:
    SparseLongitudinalFeaturesProduct(const SBaseArrayDouble2dPtrList1D &features);


    inline ulong get_feature_product_col(ulong col1,
                                         ulong col2,
                                         ulong n_cols) const;

    void sparse_features_product(ArrayULong &row,
                                 ArrayULong &col,
                                 ArrayDouble &data,
                                 ArrayULong &out_row,
                                 ArrayULong &out_col,
                                 ArrayDouble &out_data) const;

  template <class Archive>
  void serialize(Archive & ar) {
    ar(CEREAL_NVP(n_features));
  }
};

#endif //TICK_SPARSE_LONGITUDINAL_FEATURES_PRODUCT_H
