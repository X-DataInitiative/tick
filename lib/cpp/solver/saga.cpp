// License: BSD 3 clause

#include "tick/solver/saga.h"
#include "tick/base_model/model_generalized_linear.h"
#include "tick/prox/prox_separable.h"

template <class T>
TSAGA<T>::TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int seed,
                VarianceReductionMethod variance_reduction)
    : TStoSolver<T>(epoch_size, tol, rand_type, seed),
      solver_ready(false),
      ready_step_corrections(false),
      step(step),
      variance_reduction(variance_reduction) {}

template <class T>
void TSAGA<T>::set_model(std::shared_ptr<TModel<T> > model) {
  // model must be a child of ModelGeneralizedLinear
  casted_model = std::dynamic_pointer_cast<TModelGeneralizedLinear<T> >(model);
  if (casted_model == nullptr) {
    TICK_ERROR("SAGA accepts only childs of `ModelGeneralizedLinear`")
  }
  TStoSolver<T>::set_model(model);
  ready_step_corrections = false;
  solver_ready = false;
}

template <class T>
void TSAGA<T>::initialize_solver() {
  ulong n_samples = model->get_n_samples();
  gradients_memory = Array<T>(n_samples);
  gradients_memory.fill(0);
  gradients_average = Array<T>(model->get_n_coeffs());
  gradients_average.fill(0);
  solver_ready = true;
}

template <class T>
void TSAGA<T>::prepare_solve() {
  // The point where we compute the full gradient for variance reduction is the
  // new iterate obtained at the previous epoch
  if (!solver_ready) {
    initialize_solver();
  }
  next_iterate = iterate;
  if ((model->is_sparse()) && (prox->is_separable())) {
    if (!ready_step_corrections) {
      compute_step_corrections();
    }
  }
  rand_index = 0;
  if (variance_reduction == VarianceReductionMethod::Average) {
    next_iterate.init_to_zero();
  }
  if (variance_reduction == VarianceReductionMethod::Random) {
    rand_index = rand_unif(epoch_size);
  }
}

template <class T>
void TSAGA<T>::solve() {
  prepare_solve();
  bool use_intercept = model->use_intercept();
  ulong n_features = model->get_n_features();
  if ((model->is_sparse()) && (prox->is_separable())) {
    solve_sparse_proba_updates(use_intercept, n_features);
  } else {
    solve_dense(use_intercept, n_features);
  }
}

template <class T>
void TSAGA<T>::compute_step_corrections() {
  ulong n_features = model->get_n_features();
  Array<T> columns_sparsity = casted_model->get_column_sparsity_view();
  steps_correction = Array<T>(n_features);
  for (ulong j = 0; j < n_features; ++j) {
    steps_correction[j] = 1. / columns_sparsity[j];
  }
  ready_step_corrections = true;
}

template <class T>
void TSAGA<T>::solve_dense(bool use_intercept, ulong n_features) {
  ulong n_samples = model->get_n_samples();
  for (ulong t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Get the features matrix. We know that it's dense
    BaseArray<T> x_i = model->get_features(i);
    T grad_i_factor = model->grad_i_factor(i, iterate);
    T grad_i_factor_old = gradients_memory[i];
    // Update gradient memory
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong j = 0; j < n_features; ++j) {
      T x_ij = x_i._value_dense(j);
      T grad_avg_j = gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_ij + grad_avg_j);
      // Update the gradients average over seen samples
      gradients_average[j] += grad_factor_diff * x_ij / n_samples;
    }
    // deal with intercept here
    if (use_intercept) {
      iterate[n_features] -=
          step * (grad_factor_diff + gradients_average[n_features]);
      gradients_average[n_features] += grad_factor_diff / n_samples;
    }
    // Call the prox on the iterate
    prox->call(iterate, step, iterate);
    if (variance_reduction == VarianceReductionMethod::Random &&
        t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
  TStoSolver<T>::t += epoch_size;
}

template <class T>
void TSAGA<T>::solve_sparse_proba_updates(bool use_intercept,
                                          ulong n_features) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the
  // current support (non-zero values) of the sampled vector of features
  std::shared_ptr<TProxSeparable<T> > casted_prox;
  if (prox->is_separable()) {
    casted_prox = std::static_pointer_cast<TProxSeparable<T> >(prox);
  } else {
    TICK_ERROR(
        "TSAGA<T>::solve_sparse_proba_updates can be used with a separable "
        "prox only.")
  }
  ulong n_samples = model->get_n_samples();
  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Sparse features vector
    BaseArray<T> x_i = model->get_features(i);
    T grad_i_factor = model->grad_i_factor(i, iterate);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      T x_ij = x_i.data()[idx_nnz];
      T grad_avg_j = gradients_average[j];
      // Step-size correction for coordinate j
      T step_correction = steps_correction[j];
      iterate[j] -=
          step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);
      gradients_average[j] += grad_factor_diff * x_ij / n_samples;
      // Prox is separable, apply regularization on the current coordinate
      casted_prox->call_single(j, iterate, step * step_correction, iterate);
    }
    // And let's not forget to update the intercept as well. It's updated at
    // each step, so no step-correction. Note that we call the prox, in order to
    // be consistent with the dense case (in the case where the user has the
    // weird desire to to regularize the intercept)
    if (use_intercept) {
      iterate[n_features] -=
          step * (grad_factor_diff + gradients_average[n_features]);
      gradients_average[n_features] += grad_factor_diff / n_samples;
      casted_prox->call_single(n_features, iterate, step, iterate);
    }
    // Note that the average option for variance reduction with sparse data is a
    // very bad idea, but this is caught in the python class
    if (variance_reduction == VarianceReductionMethod::Random &&
        t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
  TStoSolver<T>::t += epoch_size;
}

template class DLL_PUBLIC TSAGA<double>;
template class DLL_PUBLIC TSAGA<float>;
