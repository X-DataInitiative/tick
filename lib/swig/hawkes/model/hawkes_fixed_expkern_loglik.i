// License: BSD 3 clause

%{
#include "tick/hawkes/model/hawkes_fixed_expkern_loglik.h"
%}

class ModelHawkesFixedExpKernLogLik : public Model {

 public:

  ModelHawkesFixedExpKernLogLik(const double decay,
                                const unsigned int n_cores = 1);

  void set_data(const SArrayDoublePtrList1D &timestamps, const double end_time);

  void compute_weights();

  inline unsigned long get_rand_max() const;

  double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out_grad);
  double hessian_norm(const ArrayDouble &coeffs, const ArrayDouble &vector);
  void hessian(const ArrayDouble &coeffs, ArrayDouble &out);

  double get_decay() const;
  void set_decay(double decay);

  unsigned int get_n_threads() const;
  void set_n_threads(unsigned int n_threads);

  ulong get_n_total_jumps() const;
  ulong get_n_coeffs() const;
  ulong get_n_nodes() const;
};
