//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SGD_H_
#define LIB_INCLUDE_TICK_SOLVER_SGD_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base_model/model.h"
#include "tick/prox/prox.h"

template <class T>
class DLL_PUBLIC TSGD : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;

 private:
  T step_t;
  T step;

 public:
  TSGD(ulong epoch_size = 0, T tol = 0., RandType rand_type = RandType::unif,
       T step = 0., int seed = -1);

  inline T get_step_t() const { return step_t; }

  inline T get_step() const { return step; }

  inline void set_step(T step) { this->step = step; }

  void solve();

  void solve_sparse();

  inline T get_step_t();
};

using SGD = TSGD<double>;
using SGDDouble = TSGD<double>;
using SGDFloat = TSGD<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SGD_H_
