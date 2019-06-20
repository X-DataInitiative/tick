// License: BSD 3 clause

%{
#include "tick/preprocessing/longitudinal_features_lagger.h"
%}

class LongitudinalFeaturesLagger {

 public:
  // This exists soley for cereal/swig
  LongitudinalFeaturesLagger();

  LongitudinalFeaturesLagger(ulong n_intervals,
                             SArrayULongPtr n_lags);

  void dense_lag_preprocessor(ArrayDouble2d &features,
                              ArrayDouble2d &out,
                              ulong censoring) const;

  void sparse_lag_preprocessor(ArrayULong &row, ArrayULong &col,
                               ArrayDouble &data, ArrayULong &out_row,
                               ArrayULong &out_col, ArrayDouble &out_data,
                               ulong censoring) const;
};

TICK_MAKE_PICKLABLE(LongitudinalFeaturesLagger);