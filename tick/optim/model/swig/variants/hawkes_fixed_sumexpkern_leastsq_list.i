
%{
#include "variants/hawkes_fixed_sumexpkern_leastsq_list.h"
%}


class ModelHawkesFixedSumExpKernLeastSqList : public ModelHawkesLeastSqList {
    
public:
    
  ModelHawkesFixedSumExpKernLeastSqList(const ArrayDouble &decays,
                                        const unsigned int max_n_threads = 1,
                                        const unsigned int optimization_level = 0);

  void set_decays(const ArrayDouble &decays);

  ulong get_n_decays() const;
};
