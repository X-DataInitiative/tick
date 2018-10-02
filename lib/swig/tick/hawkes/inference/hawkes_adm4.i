// License: BSD 3 clause


%include std_shared_ptr.i
%shared_ptr(HawkesADM4);

%{
#include "tick/hawkes/inference/hawkes_adm4.h"
%}

class HawkesADM4 : public ModelHawkesList {

 public:

  HawkesADM4(const double decay, const double rho, const int max_n_threads=1,
             const unsigned int optimization_level=0);

  void solve(ArrayDouble &mu, ArrayDouble2d &auv, ArrayDouble2d &z1uv, ArrayDouble2d &z2uv,
             ArrayDouble2d &u1uv, ArrayDouble2d &u2uv);

  void compute_weights();

  double get_decay() const;
  void set_decay(const double decay);
  double get_rho() const;
  void set_rho(const double rho);
};

