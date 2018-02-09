//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SDCA_H_
#define LIB_INCLUDE_TICK_SOLVER_SDCA_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base_model/model.h"

// TODO: profile the code of SDCA to check if it's faster
// TODO: code accelerated SDCA

template <class T>
class TSDCA : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::get_next_i;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::rand_max;

 public:
  using SArrayTPtr = std::shared_ptr<SArray<T>>;

 protected:
  ulong n_coeffs;

  // A boolean that attests that our arrays of ascent variables and dual
  // variables are initialized with the right size
  bool stored_variables_ready;

  // Store for coefficient update before prox call.
  Array<T> tmp_primal_vector;

  // Level of ridge regularization. This is mandatory for SDCA.
  T l_l2sq;

  // Ascent variables
  Array<T> delta;

  // The dual variable
  Array<T> dual_vector;

 public:
  explicit TSDCA(T l_l2sq, ulong epoch_size = 0, T tol = 0.,
                 RandType rand_type = RandType::unif, int seed = -1);

  void reset() override;

  void solve() override;

  void set_model(std::shared_ptr<TModel<T>> model) override;

  T get_l_l2sq() const { return l_l2sq; }

  void set_l_l2sq(T l_l2sq) { this->l_l2sq = l_l2sq; }

  SArrayTPtr get_primal_vector() const {
    Array<T> copy = iterate;
    return copy.as_sarray_ptr();
  }

  SArrayTPtr get_dual_vector() const {
    Array<T> copy = dual_vector;
    return copy.as_sarray_ptr();
  }

  void set_starting_iterate();
  void set_starting_iterate(Array<T> &dual_vector) override;

 private:
  T get_scaled_l_l2sq() const {
    // In order to solve the same problem than other solvers, we need to rescale
    // the penalty parameter if some observations are not considered in SDCA.
    // This is useful for Poisson regression with identity link
    return l_l2sq * model->get_n_samples() / rand_max;
  }
};

using SDCA = TSDCA<double>;
using SDCADouble = TSDCA<double>;
using SDCAFloat = TSDCA<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SDCA_H_
