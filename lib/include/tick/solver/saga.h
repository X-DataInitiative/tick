#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include "sto_solver.h"
#include "tick/base/base.h"
#include "tick/base_model/model_generalized_linear.h"

template <class T>
class TSAGA : public TStoSolver<T> {
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

 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

 protected:
  bool solver_ready = false;
  bool ready_step_corrections = false;
  uint64_t rand_index = 0;
  T step = 0;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<T> steps_correction;

  VarianceReductionMethod variance_reduction;

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
  TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed,
        VarianceReductionMethod variance_reduction =
            TSAGA<T>::VarianceReductionMethod::Last);

  void solve() override;

  void set_model(std::shared_ptr<TModel<T> > model) override;

  T get_step() const { return step; }

  void set_step(T step) { this->step = step; }

  TSAGA<T>::VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(
      TSAGA<T>::VarianceReductionMethod _variance_reduction) {
    variance_reduction = _variance_reduction;
  }
};

using SAGA = TSAGA<double>;
using SAGADouble = TSAGA<double>;
using SAGAFloat = TSAGA<float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
