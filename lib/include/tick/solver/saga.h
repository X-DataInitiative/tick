#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "sto_solver.h"

class SAGA : public StoSolver {
 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

 private:
  double step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  ArrayDouble steps_correction;

  VarianceReductionMethod variance_reduction;

  ArrayDouble next_iterate;

  bool solver_ready;
  // The past gradients. Can be stored in a 1D array since we consider only GLMs
  // with this solver
  ArrayDouble gradients_memory;
  ArrayDouble gradients_average;

  ulong rand_index;
  bool ready_step_corrections;

  void initialize_solver();

  void prepare_solve();

  void solve_dense(bool use_intercept, ulong n_features);

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

 public:
  SAGA(ulong epoch_size,
       double tol,
       RandType rand_type,
       double step,
       int seed = -1,
       VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last);

  void solve() override;

  void set_model(ModelPtr model) override;

  double get_step() const {
    return step;
  }

  void set_step(double step) {
    this->step = step;
  }

  VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(VarianceReductionMethod variance_reduction) {
    SAGA::variance_reduction = variance_reduction;
  }

  void set_starting_iterate(ArrayDouble &new_iterate) override;
};

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
