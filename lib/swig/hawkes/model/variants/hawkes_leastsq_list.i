// License: BSD 3 clause


%{
#include "tick/hawkes/model/variants/hawkes_leastsq_list.h"
%}

class ModelHawkesLeastSqList : public ModelHawkesList {

 public:
  ModelHawkesLeastSqList(const int max_n_threads = 1,
                         const unsigned int optimization_level = 0);

  void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

  void compute_weights();
};
