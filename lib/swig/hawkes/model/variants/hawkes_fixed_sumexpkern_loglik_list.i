// License: BSD 3 clause


%{
#include "tick/hawkes/model/variants/hawkes_fixed_sumexpkern_loglik_list.h"
%}


class ModelHawkesFixedSumExpKernLogLikList : public ModelHawkesFixedKernLogLikList {
    
public:
    
  ModelHawkesFixedSumExpKernLogLikList(const ArrayDouble &decays,
                                       const int max_n_threads = 1);

  void set_decays(ArrayDouble &decays);
  SArrayDoublePtr get_decays() const;
};
