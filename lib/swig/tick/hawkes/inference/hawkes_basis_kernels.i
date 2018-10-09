// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesBasisKernels);

%{
#include "tick/hawkes/inference/hawkes_basis_kernels.h"
%}

class HawkesBasisKernels : public ModelHawkesList {
 public :
  HawkesBasisKernels(const ulong D,
                       const double kernel_dt,
                       const double kernel_tmax,
                       const double alpha,
                       const int max_n_threads = 1);

  double solve(ArrayDouble &mu,
               ArrayDouble2d &gdm,
               ArrayDouble2d &auvd,
               ulong max_iter_gdm,
               double max_tol_gdm);

  double get_kernel_support() const;
  ulong get_kernel_size() const;
  inline double get_kernel_dt() const;
  ulong get_n_basis() const;
  double get_alpha() const;
  SArrayDoublePtr get_kernel_discretization();

  void set_kernel_support(const double kernel_support);
  void set_kernel_size(const ulong kernel_size);
  void set_kernel_dt(const double kernel_dt);
  void set_n_basis(const ulong n_basis);
  void set_alpha(const double alpha);
};

