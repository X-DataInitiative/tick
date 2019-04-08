// License: BSD 3 clause


%{
#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_loglik.h"
%}


class ModelHawkesExpKernLogLik : public ModelHawkesLogLik {
    
public:
    
  ModelHawkesExpKernLogLik(const double decay,
                                    const int max_n_threads = 1);

  void set_decay(const double decay);
};
