// License: BSD 3 clause

%{
#include "tick/preprocessing/sparse_longitudinal_features_product.h"
%}

%include serialization.i

class SparseLongitudinalFeaturesProduct {

  public:
    // This exists soley for cereal/swig
    SparseLongitudinalFeaturesProduct();

    SparseLongitudinalFeaturesProduct(const ulong n_features);

    void sparse_features_product(ArrayULong &row,
                                 ArrayULong &col,
                                 ArrayDouble &data,
                                 ArrayULong &out_row,
                                 ArrayULong &out_col,
                                 ArrayDouble &out_data) const;
};

TICK_MAKE_PICKLABLE(SparseLongitudinalFeaturesProduct);