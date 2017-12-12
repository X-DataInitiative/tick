// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesSumGaussians);

%{
#include "tick/hawkes/inference/hawkes_sumgaussians.h"
%}

class HawkesSumGaussians : public ModelHawkesList {

 public:

  HawkesSumGaussians(const ulong n_gaussians, const double max_mean_gaussian, const double step_size,
                     const double strength_lasso, const double strength_grouplasso,
                     const ulong em_max_iter, const int max_n_threads = 1,
                     const unsigned int optimization_level = 0);

  void compute_weights();

  void solve(ArrayDouble &mu, ArrayDouble2d &amplitudes);

  ulong get_n_gaussians() const;
  void set_n_gaussians(const ulong n_gaussians);
  ulong get_em_max_iter() const;
  void set_em_max_iter(const ulong em_max_iter);
  double get_max_mean_gaussian() const;
  void set_max_mean_gaussian(const double max_mean_gaussian);
  double get_step_size() const;
  void set_step_size(const double step_size);
  double get_strength_lasso() const;
  void set_strength_lasso(const double strength_lasso);
  double get_strength_grouplasso() const;
  void set_strength_grouplasso(const double strength_grouplasso);
};