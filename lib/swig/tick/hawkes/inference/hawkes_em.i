// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesEM);

%{
#include "tick/hawkes/inference/hawkes_em.h"
%}

class HawkesEM : public ModelHawkesList {
 public :
  HawkesEM(const double kernel_support, const ulong kernel_size,
           const int max_n_threads = 1);

  HawkesEM(const SArrayDoublePtr kernel_discretization, const int max_n_threads = 1);

  void allocate_weights();

  void solve(ArrayDouble &mu, ArrayDouble2d &kernels);

  SArrayDouble2dPtr get_kernel_norms(ArrayDouble2d &kernels) const;
  double loglikelihood(ArrayDouble &mu, ArrayDouble2d &kernels);

  double get_kernel_support() const;
  ulong get_kernel_size() const;
  double get_kernel_fixed_dt() const;
  SArrayDoublePtr get_kernel_discretization() const;

  void set_kernel_support(const double kernel_support);
  void set_kernel_size(const ulong kernel_size);
  void set_kernel_dt(const double kernel_dt);
  void set_kernel_discretization(const SArrayDoublePtr kernel_discretization);
};
