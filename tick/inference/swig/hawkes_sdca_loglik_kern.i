
// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesSDCALoglikKern);

%{
#include "hawkes_sdca_loglik_kern.h"
%}

class HawkesSDCALoglikKern : public ModelHawkesList {
 public:
  HawkesSDCALoglikKern(double decay, double l_l2sq,
                       int max_n_threads = 1, double tol = 0.,
                       RandType rand_type = RandType::unif, int seed = -1);

  void compute_weights();

  void solve();

  double get_decay() const;
  void set_decay(double decay);

  SArrayDoublePtr get_iterate();
};