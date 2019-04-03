#ifndef LIB_INCLUDE_TICK_SOLVER_SVRG_H_
#define LIB_INCLUDE_TICK_SOLVER_SVRG_H_

// License: BSD 3 clause

#include "sgd.h"
#include "tick/array/array.h"
#include "tick/prox/prox.h"
#include "tick/prox/prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TSVRG : public TStoSolver<T, K> {
 protected:
  using TStoSolver<T, K>::t;
  using TStoSolver<T, K>::model;
  using TStoSolver<T, K>::iterate;
  using TStoSolver<T, K>::prox;
  using TStoSolver<T, K>::epoch_size;
  using TStoSolver<T, K>::get_next_i;
  using TStoSolver<T, K>::rand_unif;
  using TStoSolver<T, K>::get_generator;

 public:
  using TStoSolver<T, K>::get_class_name;
  using TStoSolver<T, K>::solve;

 private:
  unsigned int n_threads = 1;
  T step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  SVRG_VarianceReductionMethod variance_reduction;

  Array<T> full_gradient;
  Array<K> fixed_w;
  Array<T> grad_i;
  Array<T> grad_i_fixed_w;
  Array<T> next_iterate;

  ulong rand_index;
  bool ready_step_corrections;
  SVRG_StepType step_type;

  std::shared_ptr<TProxSeparable<T, K>> casted_prox;

  void prepare_solve();

  void solve_dense();

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

  void dense_single_thread_solver(const ulong& next_i);

  void sparse_single_thread_solver(const ulong& next_i, const ulong& n_features,
                                   const bool use_intercept);

 protected:
  // if K == std::atomic<T> (Wild case)
  template<class T1 = T, class K1 = K>
  typename std::enable_if<std::is_same<T1, K1>::value, void>::type
  update_iterate(ulong j, T x_ij, T grad_factor_diff, T step_correction, T full_gradient_j);

  // if K == std::atomic<T> (Atomic case)
  template<class T1 = T, class K1 = K>
  typename std::enable_if<std::is_same<std::atomic<T1>, K1>::value, void>::type
  update_iterate(ulong j, T x_ij, T grad_factor_diff, T step_correction, T full_gradient_j);

 public:
  // This exists soley for cereal/swig
  TSVRG() : TSVRG<T, K>(0, 0, RandType::unif, 0) {}

  TSVRG(ulong epoch_size, T tol, RandType rand_type, T step, size_t record_every = 1, int seed = -1,
        unsigned int n_threads = 1,
        SVRG_VarianceReductionMethod variance_reduction = SVRG_VarianceReductionMethod::Last,
        SVRG_StepType step_method = SVRG_StepType::Fixed);

  void solve_one_epoch() override;

  void set_model(std::shared_ptr<TModel<T, K>> model) override;

  T get_step() const { return step; }

  void set_step(T step) { TSVRG<T, K>::step = step; }

  SVRG_VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(SVRG_VarianceReductionMethod variance_reduction) {
    TSVRG<T, K>::variance_reduction = variance_reduction;
  }

  SVRG_StepType get_step_type() { return step_type; }

  void set_step_type(SVRG_StepType step_type) {
    TSVRG<T, K>::step_type = step_type;
  }

  void set_starting_iterate(Array<T>& new_iterate) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("StoSolver",
                        cereal::base_class<TStoSolver<T, K>>(this)));

    ar(CEREAL_NVP(step));
    ar(CEREAL_NVP(steps_correction));
    ar(CEREAL_NVP(variance_reduction));
    ar(CEREAL_NVP(full_gradient));
    ar(CEREAL_NVP(fixed_w));
    ar(CEREAL_NVP(grad_i));
    ar(CEREAL_NVP(grad_i_fixed_w));
    ar(CEREAL_NVP(next_iterate));
    ar(CEREAL_NVP(ready_step_corrections));
    ar(CEREAL_NVP(step_type));
  }

  BoolStrReport compare(const TSVRG<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool are_equal =
        TStoSolver<T, K>::compare(that, ss) && TICK_CMP_REPORT(ss, step) &&
        TICK_CMP_REPORT(ss, steps_correction) &&
        TICK_CMP_REPORT(ss, variance_reduction) &&
        TICK_CMP_REPORT(ss, full_gradient) && TICK_CMP_REPORT(ss, fixed_w) &&
        TICK_CMP_REPORT(ss, grad_i) && TICK_CMP_REPORT(ss, grad_i_fixed_w) &&
        TICK_CMP_REPORT(ss, next_iterate) &&
        TICK_CMP_REPORT(ss, ready_step_corrections) &&
        TICK_CMP_REPORT(ss, step_type);
    return BoolStrReport(are_equal, ss.str());
  }

  BoolStrReport operator==(const TSVRG<T, K>& that) { return compare(that); }
};

using SVRGDouble = TSVRG<double, double>;
using SVRGDoubleVector = std::vector<SVRGDouble>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SVRGDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SVRGDouble)

using SVRGFloat = TSVRG<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SVRGFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SVRGFloat)

using SVRGDoubleAtomicIterate = TSVRG<double, std::atomic<double>>;
using SVRGFloatAtomicIterate = TSVRG<float, std::atomic<float>>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SVRG_H_
