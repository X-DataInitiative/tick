// License: BSD 3 clause


%{
#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_leastsq.h"
%}


class ModelHawkesSumExpKernLeastSq : public ModelHawkesLeastSq {
    
public:
  ModelHawkesSumExpKernLeastSq();
  ModelHawkesSumExpKernLeastSq(const ArrayDouble &decays,
                                        const ulong n_baselines,
                                        const double period_length,
                                        const unsigned int max_n_threads = 1,
                                        const unsigned int optimization_level = 0);
  void hessian(ArrayDouble &out);

  void set_decays(const ArrayDouble &decays);

  ulong get_n_decays() const;

  ulong get_n_baselines() const;
  double get_period_length() const;

  void set_n_baselines(ulong n_baselines);
  void set_period_length(double period_length);

  void compute_penalization_constant(double x, ArrayDouble &pen_mu , ArrayDouble &pen_L1_alpha,
                                         double pen_mu_const1, double pen_mu_const2,
                                         double pen_L1_const1, double pen_L1_const2,
                                         double normalization);
};

TICK_MAKE_PICKLABLE(ModelHawkesSumExpKernLeastSq);