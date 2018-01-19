//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef LIB_INCLUDE_TICK_SOLVER_SDCA_H_
#define LIB_INCLUDE_TICK_SOLVER_SDCA_H_

// License: BSD 3 clause

#include "tick/base_model/model.h"
#include "sto_solver.h"


// TODO: profile the code of SDCA to check if it's faster

// TODO: code accelerated SDCA


class SDCA : public StoSolver {
 protected:
  ulong n_coeffs;

  // A boolean that attests that our arrays of ascent variables and dual variables are initialized
  // with the right size
  bool stored_variables_ready;

  // Store for coefficient update before prox call.
  ArrayDouble tmp_primal_vector;

  // Level of ridge regularization. This is mandatory for SDCA.
  double l_l2sq;

  // Ascent variables
  ArrayDouble delta;

  // The dual variable
  ArrayDouble dual_vector;

 public:
  explicit SDCA(double l_l2sq,
                ulong epoch_size = 0,
                double tol = 0.,
                RandType rand_type = RandType::unif,
                int seed = -1);

  void reset() override;

  void solve() override;

  void set_model(ModelPtr model) override;

  double get_l_l2sq() const {
    return l_l2sq;
  }

  void set_l_l2sq(double l_l2sq) {
    this->l_l2sq = l_l2sq;
  }

  SArrayDoublePtr get_primal_vector() const {
    ArrayDouble copy = iterate;
    return copy.as_sarray_ptr();
  }

  SArrayDoublePtr get_dual_vector() const {
    ArrayDouble copy = dual_vector;
    return copy.as_sarray_ptr();
  }

  void set_starting_iterate();
  void set_starting_iterate(ArrayDouble &dual_vector) override;

 private:
  double get_scaled_l_l2sq() const {
    // In order to solve the same problem than other solvers, we need to rescale the penalty
    // parameter if some observations are not considered in SDCA. This is useful for
    // Poisson regression with identity link
    return l_l2sq * model->get_n_samples() / rand_max;
  }
};

#endif  // LIB_INCLUDE_TICK_SOLVER_SDCA_H_
