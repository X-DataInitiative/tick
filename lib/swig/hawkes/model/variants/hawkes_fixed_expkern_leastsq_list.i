// License: BSD 3 clause


%{
#include "tick/hawkes/model/variants/hawkes_fixed_expkern_leastsq_list.h"
%}


class ModelHawkesFixedExpKernLeastSqList : public ModelHawkesLeastSqList {
    
public:
  ModelHawkesFixedExpKernLeastSqList();
  ModelHawkesFixedExpKernLeastSqList(const SArrayDouble2dPtr decays,
                                     const int max_n_threads = 1,
                                     const unsigned int optimization_level = 0);

  void hessian(ArrayDouble &out);
  void set_decays(const SArrayDouble2dPtr decays);
};

TICK_MAKE_PICKLABLE(ModelHawkesFixedExpKernLeastSqList);