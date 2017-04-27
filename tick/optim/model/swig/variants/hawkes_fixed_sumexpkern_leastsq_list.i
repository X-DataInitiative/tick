
%{
#include "variants/hawkes_fixed_sumexpkern_leastsq_list.h"
%}


class ModelHawkesFixedSumExpKernLeastSqList : public ModelHawkesLeastSqList {
    
public:
    
  ModelHawkesFixedSumExpKernLeastSqList(const ArrayDouble &decays,
                                        const ulong n_baselines,
                                        const double period_length,
                                        const unsigned int max_n_threads = 1,
                                        const unsigned int optimization_level = 0);

  void set_decays(const ArrayDouble &decays);

  ulong get_n_decays() const;

  ulong get_n_baselines() const;
  double get_period_length() const;

  void set_n_baselines(ulong n_baselines);
  void set_period_length(double period_length);
};
