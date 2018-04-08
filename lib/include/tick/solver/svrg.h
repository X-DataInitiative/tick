#ifndef LIB_INCLUDE_TICK_SOLVER_SVRG_H_
#define LIB_INCLUDE_TICK_SOLVER_SVRG_H_

// License: BSD 3 clause

#include "sgd.h"
#include "tick/array/array.h"
#include "tick/prox/prox.h"
#include "tick/prox/prox_separable.h"

template <class T>
class DLL_PUBLIC TSVRG : public TStoSolver<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;
  using TStoSolver<T>::rand_unif;

 public:
  using TStoSolver<T>::get_class_name;

 private:
  int n_threads = 1;
  T step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  SVRG_VarianceReductionMethod variance_reduction;

  Array<T> full_gradient;
  Array<T> fixed_w;
  Array<T> grad_i;
  Array<T> grad_i_fixed_w;
  Array<T> next_iterate;

  ulong rand_index;
  bool ready_step_corrections;
  SVRG_StepType step_type;

  void prepare_solve();

  void solve_dense();

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

  void dense_single_thread_solver(const ulong& next_i);

  // TProxSeparable<T>* is a raw pointer here as the
  //  ownership of the pointer is handled by
  //  a shared_ptr which is above it in the same
  //  scope so a shared_ptr is not needed
  void sparse_single_thread_solver(const ulong& next_i, const ulong& n_features,
                                   const bool use_intercept,
                                   TProxSeparable<T>*& casted_prox);

 public:
  // This exists soley for cereal/swig
  TSVRG() : TSVRG<T>(0, 0, RandType::unif, 0) {}

  TSVRG(ulong epoch_size, T tol, RandType rand_type, T step, int seed = -1,
        int n_threads = 1,
        SVRG_VarianceReductionMethod variance_reduction =
            SVRG_VarianceReductionMethod::Last,
        SVRG_StepType step_method = SVRG_StepType::Fixed);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T> > model) override;

  T get_step() const { return step; }

  void set_step(T step) { TSVRG<T>::step = step; }

  SVRG_VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(SVRG_VarianceReductionMethod variance_reduction) {
    TSVRG<T>::variance_reduction = variance_reduction;
  }

  SVRG_StepType get_step_type() { return step_type; }

  void set_step_type(SVRG_StepType step_type) {
    TSVRG<T>::step_type = step_type;
  }

  void set_starting_iterate(Array<T>& new_iterate) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("StoSolver", cereal::base_class<TStoSolver<T>>(this)));

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

  BoolStrReport compare(const TSVRG<T>& that) {
    std::stringstream ss;
    ss << get_class_name() << std::endl;
    bool are_equal =
        TStoSolver<T>::compare(that, ss) && TICK_CMP_REPORT(ss, step) &&
        TICK_CMP_REPORT(ss, steps_correction) &&
        TICK_CMP_REPORT(ss, variance_reduction) &&
        TICK_CMP_REPORT(ss, full_gradient) && TICK_CMP_REPORT(ss, fixed_w) &&
        TICK_CMP_REPORT(ss, grad_i) && TICK_CMP_REPORT(ss, grad_i_fixed_w) &&
        TICK_CMP_REPORT(ss, next_iterate) &&
        TICK_CMP_REPORT(ss, ready_step_corrections) &&
        TICK_CMP_REPORT(ss, step_type);
    return BoolStrReport(are_equal, ss.str());
  }

  BoolStrReport operator==(const TSVRG<T>& that) { return compare(that); }

  static std::shared_ptr<TSVRG<T>> AS_NULL() {
    return std::move(std::shared_ptr<TSVRG<T>>(new TSVRG<T>));
  }
};

using SVRG = TSVRG<double>;
using SVRGDouble = TSVRG<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SVRGDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SVRGDouble)

using SVRGFloat = TSVRG<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SVRGFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(SVRGFloat)

#endif  // LIB_INCLUDE_TICK_SOLVER_SVRG_H_
