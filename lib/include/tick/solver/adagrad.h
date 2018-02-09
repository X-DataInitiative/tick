#ifndef LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
#define LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_

// License: BSD 3 clause

#include "sto_solver.h"

template <class T>
class TAdaGrad : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;

 private:
  Array<T> hist_grad;
  T step;

 public:
  TAdaGrad(ulong epoch_size, T tol, RandType rand_type, T step, int seed);

  void solve() override;

  void set_starting_iterate(Array<T> &new_iterate) override;
};

using AdaGrad = TAdaGrad<double>;
using AdaGradDouble = TAdaGrad<double>;
using AdaGradFloat = TAdaGrad<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_ADAGRAD_H_
