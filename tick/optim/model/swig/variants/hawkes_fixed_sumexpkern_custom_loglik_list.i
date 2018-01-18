// License: BSD 3 clause


%{
#include "variants/hawkes_fixed_sumexpkern_custom_loglik_list.h"
%}


class ModelHawkesFixedSumExpKernCustomLogLikList : public ModelHawkesFixedKernLogLikList {
    
public:
    
  ModelHawkesFixedSumExpKernCustomLogLikList(const ArrayDouble &decays, const ulong _MaxN_of_f,
                                       const int max_n_threads = 1);

  void set_decays(ArrayDouble &decays);
  SArrayDoublePtr get_decays() const;
  ulong get_MaxN_of_f() const;
  void set_MaxN_of_f(ulong _MaxN_of_f);
};
