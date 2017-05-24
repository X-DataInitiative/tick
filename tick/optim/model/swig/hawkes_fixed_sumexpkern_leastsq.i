

%{
#include "hawkes_fixed_sumexpkern_leastsq.h"
%}


class ModelHawkesFixedSumExpKernLeastSq : public Model {

 public:

  ModelHawkesFixedSumExpKernLeastSq(const ArrayDouble &decays,
                                    const ulong n_baselines,
                                    const double period_length,
                                    const unsigned int max_n_threads = 1,
                                    const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList1D &timestamps, const double end_time);

  void compute_weights();

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

  ulong get_n_total_jumps() const;
  ulong get_n_coeffs() const;
  ulong get_n_nodes() const;

  ulong get_n_baselines() const;
  double get_period_length() const;

  void set_n_baselines(ulong n_baselines);
  void set_period_length(double period_length);
};