// License: BSD 3 clause



%{
#include "tick/hawkes/model/hawkes_fixed_expkern_leastsq.h"
%}


class ModelHawkesFixedExpKernLeastSq : public Model {

 public:

  ModelHawkesFixedExpKernLeastSq(const SArrayDouble2dPtr decays,
                                 const int max_n_threads = 1,
                                 const unsigned int optimization_level = 0);

  void set_data(const SArrayDoublePtrList1D &timestamps, const double end_time);
  void set_decays(const SArrayDouble2dPtr decays);
  void compute_weights();

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);
  void hessian(ArrayDouble &out);

  ulong get_n_total_jumps();
  ulong get_n_coeffs() const;
  ulong get_n_nodes() const;
};