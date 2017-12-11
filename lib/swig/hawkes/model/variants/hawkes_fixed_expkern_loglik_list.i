// License: BSD 3 clause


%{
#include "tick/hawkes/model/variants/hawkes_fixed_expkern_loglik_list.h"
%}


class ModelHawkesFixedExpKernLogLikList : public ModelHawkesFixedKernLogLikList {
    
public:
    
  ModelHawkesFixedExpKernLogLikList(const double decay,
                                    const int max_n_threads = 1);

  void set_decay(const double decay);
};
