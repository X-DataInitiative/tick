// License: BSD 3 clause

%{
#include "sparse_longitudinal_features_product.h"
%}

class SparseLongitudinalFeaturesProduct {

  public:
    SparseLongitudinalFeaturesProduct(const SBaseArrayDouble2dPtrList1D &features);

    void sparse_features_product(ArrayULong &row,
                                 ArrayULong &col,
                                 ArrayDouble &data,
                                 ArrayULong &out_row,
                                 ArrayULong &out_col,
                                 ArrayDouble &out_data) const;
};

TICK_MAKE_PICKLABLE(SparseLongitudinalFeaturesProduct);