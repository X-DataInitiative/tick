#ifndef TICK_OPTIM_SOLVER_SRC_SAGA_H_
#define TICK_OPTIM_SOLVER_SRC_SAGA_H_

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
  // Empty constructor only used for serialization
  SAGA(): StoSolver(-1) {};

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

 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<StoSolver>(this)));

    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(variance_reduction));
    ar(CEREAL_NVP(next_iterate));
    ar(CEREAL_NVP(solver_ready));
    ar(CEREAL_NVP(gradients_memory));
    ar(CEREAL_NVP(gradients_average));
    ar(CEREAL_NVP(rand_index));
    ar(CEREAL_NVP(ready_step_corrections));
  }

};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGA, cereal::specialization::member_serialize);

#endif  // TICK_OPTIM_SOLVER_SRC_SAGA_H_
