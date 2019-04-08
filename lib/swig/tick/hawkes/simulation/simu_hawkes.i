// License: BSD 3 clause


%{
#include "tick/hawkes/simulation/simu_hawkes.h"
%}


class Hawkes : public PP {
 public :

  Hawkes(int dimension, int seed = -1);

  void set_kernel(unsigned int i, unsigned int j, std::shared_ptr<HawkesKernel> kernel);

  void set_baseline(unsigned int i, double baseline);
  void set_baseline(unsigned int i, ArrayDouble &times, ArrayDouble &values);
  void set_baseline(unsigned int i, TimeFunction time_function);

  SArrayDoublePtr get_baseline(unsigned int i, ArrayDouble &t);
  double get_baseline(unsigned int i, double t);
};

TICK_MAKE_PICKLABLE(Hawkes, 0);
