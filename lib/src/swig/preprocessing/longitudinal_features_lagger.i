// License: BSD 3 clause

%{
#include "tick/preprocessing/longitudinal_features_lagger.h"
%}

class LongitudinalFeaturesLagger {

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

};

TICK_MAKE_PICKLABLE(LongitudinalFeaturesLagger);