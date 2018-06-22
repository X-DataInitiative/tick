#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base/base.h"
#include "tick/base_model/model_generalized_linear.h"

template <class T>
class DLL_PUBLIC TSAGA : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;
  using TStoSolver<T>::rand_unif;

 public:
  using TStoSolver<T>::set_model;
  using TStoSolver<T>::get_minimizer;
  using TStoSolver<T>::set_starting_iterate;
  using TStoSolver<T>::get_class_name;

 protected:
  bool solver_ready = false;
  bool ready_step_corrections = false;
  uint64_t rand_index = 0;
  T step = 0;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  SAGA_VarianceReductionMethod variance_reduction;

  Array<T> next_iterate;

  Array<T> gradients_memory;
  Array<T> gradients_average;

  std::shared_ptr<TModelGeneralizedLinear<T> > casted_model;

  void initialize_solver();

  void prepare_solve();

  void solve_dense(bool use_intercept, ulong n_features);

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

  void set_starting_iterate(Array<T> &new_iterate) override;

 public:
  // This exists soley for cereal/swig
  TSAGA() : TSAGA<T>(0, 0, RandType::unif, 0, 0) {}

  TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed,
        SAGA_VarianceReductionMethod variance_reduction =
            SAGA_VarianceReductionMethod::Last);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T> > model) override;

  T get_step() const { return step; }

  void set_step(T step) { this->step = step; }

  SAGA_VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(
      SAGA_VarianceReductionMethod _variance_reduction) {
    variance_reduction = _variance_reduction;
  }

 public:
  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<TStoSolver<T> >(this)));

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

  BoolStrReport compare(const TSAGA<T> &that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool ret = TStoSolver<T>::compare(that, ss) && TICK_CMP_REPORT(ss, step) &&
               TICK_CMP_REPORT(ss, steps_correction) &&
               TICK_CMP_REPORT(ss, variance_reduction) &&
               TICK_CMP_REPORT(ss, next_iterate) &&
               TICK_CMP_REPORT(ss, solver_ready) &&
               TICK_CMP_REPORT(ss, gradients_memory) &&
               TICK_CMP_REPORT(ss, gradients_average) &&
               TICK_CMP_REPORT(ss, rand_index) &&
               TICK_CMP_REPORT(ss, ready_step_corrections);
    return BoolStrReport(ret, ss.str());
  }

  BoolStrReport operator==(const TSAGA<T> &that) { return compare(that); }

  static std::shared_ptr<TSAGA<T> > AS_NULL() {
    return std::move(std::shared_ptr<TSAGA<T> >(new TSAGA<T>));
  }
};

using SAGA = TSAGA<double>;

using SAGADouble = TSAGA<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGADouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGADouble)

using SAGAFloat = TSAGA<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SAGAFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SAGAFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
