#ifndef LIB_INCLUDE_TICK_SOLVER_SAGA_H_
#define LIB_INCLUDE_TICK_SOLVER_SAGA_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "sto_solver.h"
#include "tick/base_model/model_generalized_linear.h"

template <class T, class K>
class BaseSAGA : public TStoSolver<T, K> {
 protected:
  using TStoSolver<T, K>::model;
  using TStoSolver<T, K>::iterate;
  using TStoSolver<T, K>::prox;
  using TStoSolver<T, K>::epoch_size;
  using TStoSolver<T, K>::get_next_i;
  using TStoSolver<T, K>::rand_unif;

 public:
  using TStoSolver<T, K>::set_model;
  using TStoSolver<T, K>::get_minimizer;
  using TStoSolver<T, K>::set_starting_iterate;

 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

 protected:
  bool solver_ready = 0;
  bool ready_step_corrections = 0;
  uint64_t rand_index = 0;
  double step = 0;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  Array<K> steps_correction;

  VarianceReductionMethod variance_reduction;

  Array<T> next_iterate;

  Array<T> gradients_memory;
  Array<T> gradients_average;

  std::shared_ptr<TModelGeneralizedLinear<T, K> > casted_model;

  void initialize_solver();

  void prepare_solve();

  void solve_dense(bool use_intercept, ulong n_features);

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

  virtual void decrement_iterate_j_dense(
    size_t j,
    K step, K x_ij, K grad_factor_diff, K grad_avg_j);

  virtual void decrement_iterate_j_sparse(
    size_t j,
    K step, K x_ij, K grad_factor_diff, K grad_avg_j);

  void set_starting_iterate(Array<T> &new_iterate) override;

 public:
  BaseSAGA(
    ulong epoch_size,
    double tol,
    RandType rand_type,
    double step,
    int seed,
    VarianceReductionMethod variance_reduction
    = BaseSAGA<T, K>::VarianceReductionMethod::Last);

  virtual void solve();

  void set_model(std::shared_ptr<TModel<T, K> > model) override;

  double get_step() const {
    return step;
  }

  void set_step(double step) {
    this->step = step;
  }

  BaseSAGA<T, K>::VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  void set_variance_reduction(BaseSAGA<T, K>::VarianceReductionMethod _variance_reduction) {
    variance_reduction = _variance_reduction;
  }
};

//####### Template specializations

// iterate[j] -= step * (grad_factor_diff * x_ij + grad_avg_j);
template <class T, class K>
void BaseSAGA<T, K>::decrement_iterate_j_dense(
  size_t j,
  K step,
  K x_ij,
  K grad_factor_diff,
  K grad_avg_j
) {
  iterate[j] -= step * (grad_factor_diff * x_ij + grad_avg_j);
}


// step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);
template <class T, class K>
void BaseSAGA<T, K>::decrement_iterate_j_sparse(
  size_t j,
  K step,
  K x_ij,
  K grad_factor_diff,
  K grad_avg_j
) {
  K step_correction = steps_correction[j];
  iterate[j] -= step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);
}

//####### Template specializations

class SAGA : public BaseSAGA<double, double> {
 public:
  using BaseSAGA<double, double>::set_variance_reduction;
  using BaseSAGA<double, double>::set_starting_iterate;
  using BaseSAGA<double, double>::get_minimizer;

 protected:
  using BaseSAGA<double, double>::decrement_iterate_j_dense;
  using BaseSAGA<double, double>::decrement_iterate_j_sparse;

 public:
  SAGA(ulong epoch_size,
       double tol,
       RandType rand_type,
       double step,
       int seed = -1,
       BaseSAGA<double, double>::VarianceReductionMethod variance_reduction
        = BaseSAGA<double, double>::VarianceReductionMethod::Last);
};

using BaseSAGADouble = BaseSAGA<double, double>;
using BaseSAGAFloat  = BaseSAGA<float , float>;

#endif  // LIB_INCLUDE_TICK_SOLVER_SAGA_H_
