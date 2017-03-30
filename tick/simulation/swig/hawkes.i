
%{
#include "hawkes.h"
%}


class Hawkes : public PP {
 public :

  Hawkes(int dimension, int seed = -1);

  void set_kernel(int i,int j, std::shared_ptr<HawkesKernel> kernel);

  void set_mu(int i, HawkesMuPtr mu);
  void set_mu(int i, double mu);
  double get_mu(int i);
};

TICK_MAKE_PICKLABLE(Hawkes, 0);
