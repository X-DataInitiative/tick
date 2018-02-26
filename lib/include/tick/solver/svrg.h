#ifndef LIB_INCLUDE_TICK_SOLVER_SVRG_H_
#define LIB_INCLUDE_TICK_SOLVER_SVRG_H_

// License: BSD 3 clause

#include "sgd.h"
#include "tick/array/array.h"
#include "tick/prox/prox.h"
#include "tick/prox/prox_separable.h"

template <class T>
class DLL_PUBLIC TSVRG : public TStoSolver<T> {
 protected:
  using TStoSolver<T>::t;
  using TStoSolver<T>::model;
  using TStoSolver<T>::iterate;
  using TStoSolver<T>::prox;
  using TStoSolver<T>::epoch_size;
  using TStoSolver<T>::get_next_i;
  using TStoSolver<T>::rand_unif;

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
  T step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  VarianceReductionMethod variance_reduction;

  Array<T> full_gradient;
  Array<T> fixed_w;
  Array<T> grad_i;
  Array<T> grad_i_fixed_w;
  Array<T> next_iterate;

  ulong rand_index;
  bool ready_step_corrections;
  StepType step_type;

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
  TSVRG(ulong epoch_size, T tol, RandType rand_type, T step, int seed = -1,
        int n_threads = 1,
        VarianceReductionMethod variance_reduction =
            VarianceReductionMethod::Last,
        StepType step_method = StepType::Fixed);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T> > model) override;

  T get_step() const { return step; }

  void set_step(T step) { TSVRG<T>::step = step; }

  VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(VarianceReductionMethod variance_reduction) {
    TSVRG<T>::variance_reduction = variance_reduction;
  }

  StepType get_step_type() { return step_type; }
  void set_step_type(StepType step_type) { TSVRG<T>::step_type = step_type; }

  void set_starting_iterate(Array<T>& new_iterate) override;
};

using SVRG = TSVRG<double>;

using SVRGDouble = TSVRG<double>;
using SVRGFloat = TSVRG<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SVRG_H_
