// License: BSD 3 clause


%{
#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_leastsq.h"
%}


class ModelHawkesExpKernLeastSq : public ModelHawkesLeastSq {
    
public:
  ModelHawkesExpKernLeastSq();
  ModelHawkesExpKernLeastSq(const SArrayDouble2dPtr decays,
                                     const int max_n_threads = 1,
                                     const unsigned int optimization_level = 0);

  void hessian(ArrayDouble &out);
  void set_decays(const SArrayDouble2dPtr decays);
};

TICK_MAKE_PICKLABLE(ModelHawkesExpKernLeastSq);