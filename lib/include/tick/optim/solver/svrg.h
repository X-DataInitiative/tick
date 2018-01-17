#ifndef TICK_OPTIM_SOLVER_SRC_SVRG_H_
#define TICK_OPTIM_SOLVER_SRC_SVRG_H_

// License: BSD 3 clause

#include "tick/array/array.h"
#include "sgd.h"
#include "tick/optim/prox/prox.h"
#include "tick/optim/prox/prox_separable.h"

class SVRG : public StoSolver {
 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

  enum class StepType {
      Fixed = 1,
      BarzilaiBorwein = 2,
  };

 private:
  int n_threads = 1;
  double step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  ArrayDouble steps_correction;

  VarianceReductionMethod variance_reduction;
  StepType step_type;

  ArrayDouble full_gradient;
  ArrayDouble fixed_w;
  ArrayDouble grad_i;
  ArrayDouble grad_i_fixed_w;
  ArrayDouble next_iterate;

  ulong rand_index;
  bool ready_step_corrections;

  void prepare_solve();

  void solve_dense();

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

  void dense_single_thread_solver(const ulong& next_i);

  // ProxSeparable* is a raw pointer here as the
  //  ownership of the pointer is handled by
  //  a shared_ptr which is above it in the same
  //  scope so a shared_ptr is not needed
  void sparse_single_thread_solver(
      const ulong& next_i,
      const ulong& n_features,
      const bool use_intercept,
      ProxSeparable*& casted_prox);

 public:
  // Empty constructor only used for serialization
  SVRG(): StoSolver(-1) {};
  // SVRG() = default;

  SVRG(ulong epoch_size,
       double tol,
       RandType rand_type,
       double step,
       int seed = -1,
       int n_threads = 1,
       VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
       StepType step_method = StepType::Fixed);

  void solve() override;

  void set_model(ModelPtr model) override;

  double get_step() const {
    return step;
  }

  void set_step(double step) {
    SVRG::step = step;
  }

  VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(VarianceReductionMethod variance_reduction) {
    SVRG::variance_reduction = variance_reduction;
  }

  StepType get_step_type() {
    return step_type;
  }
  void set_step_type(StepType step_type) {
    SVRG::step_type = step_type;
  }

  void set_starting_iterate(ArrayDouble &new_iterate) override;


 public:
  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<StoSolver>(this)));

    ar(CEREAL_NVP(n_threads));
    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(variance_reduction));
    ar(CEREAL_NVP(step_type));
    ar(CEREAL_NVP(full_gradient));
    ar(CEREAL_NVP(fixed_w));
    ar(CEREAL_NVP(grad_i));
    ar(CEREAL_NVP(grad_i_fixed_w));
    ar(CEREAL_NVP(next_iterate));
    ar(CEREAL_NVP(rand_index));
    ar(CEREAL_NVP(ready_step_corrections));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SVRG, cereal::specialization::member_serialize);

#endif  // TICK_OPTIM_SOLVER_SRC_SVRG_H_
