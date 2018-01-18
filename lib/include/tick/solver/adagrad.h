#ifndef LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
#define LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_

// License: BSD 3 clause

#include "sto_solver.h"

class AdaGrad : public StoSolver {
 private:
  ArrayDouble hist_grad;
  double step;

 public:
  AdaGrad(ulong epoch_size, double tol, RandType rand_type, double step, int seed);

  void solve() override;

  void set_starting_iterate(ArrayDouble &new_iterate) override;
};

#endif  // LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
