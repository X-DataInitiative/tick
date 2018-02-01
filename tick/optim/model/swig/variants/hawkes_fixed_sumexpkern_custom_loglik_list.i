// License: BSD 3 clause


%{
#include "variants/hawkes_fixed_sumexpkern_custom_loglik_list.h"
%}

%shared_ptr(ModelHawkesFixedSumExpKernCustomLogLikList);


class ModelHawkesFixedSumExpKernCustomLogLikList : public ModelHawkesFixedKernLogLikList {
    
public:
    
  ModelHawkesFixedSumExpKernCustomLogLikList(const ArrayDouble &decays, const ulong _MaxN_of_f,
                                       const int max_n_threads = 1);

  void set_decays(ArrayDouble &decays);
  SArrayDoublePtr get_decays() const;
  ulong get_MaxN_of_f() const;
  void set_MaxN_of_f(ulong _MaxN_of_f);

  void set_data(const SArrayDoublePtrList2D &timestamps_list, const SArrayLongPtrList1D &global_n_list,
                                                              const VArrayDoublePtr end_times);
};
