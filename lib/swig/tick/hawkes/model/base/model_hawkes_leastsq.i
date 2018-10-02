// License: BSD 3 clause


%{
#include "tick/hawkes/model/base/model_hawkes_leastsq.h"
%}

class ModelHawkesLeastSq : public ModelHawkesList {

 public:
  ModelHawkesLeastSq(const int max_n_threads = 1,
                         const unsigned int optimization_level = 0);

  void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

  void compute_weights();
};
