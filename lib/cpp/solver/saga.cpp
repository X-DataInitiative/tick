// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/prox/prox_separable.h"
#include "tick/solver/saga.h"

SAGA::SAGA(ulong epoch_size,
           double tol,
           RandType rand_type,
           double step,
           int seed,
           VarianceReductionMethod variance_reduction)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step), variance_reduction(variance_reduction),
      solver_ready(false),
      ready_step_corrections(false) {}

void SAGA::set_model(ModelPtr model) {
  // model must be a child of ModelGeneralizedLinear
  auto casted_model = std::dynamic_pointer_cast<ModelGeneralizedLinear>(model);
  if (casted_model == nullptr) {
    TICK_ERROR("SAGA accepts only childs of `ModelGeneralizedLinear`")
  }
  StoSolver::set_model(model);
  ready_step_corrections = false;
  solver_ready = false;
}

void SAGA::initialize_solver() {
  ulong n_samples = model->get_n_samples();
  gradients_memory = ArrayDouble(n_samples);
  gradients_memory.fill(0);
  gradients_average = ArrayDouble(model->get_n_coeffs());
  gradients_average.fill(0);
  solver_ready = true;
}

void SAGA::prepare_solve() {
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

void SAGA::solve() {
  prepare_solve();
  bool use_intercept = model->use_intercept();
  ulong n_features = model->get_n_features();
  if ((model->is_sparse()) && (prox->is_separable())) {
    solve_sparse_proba_updates(use_intercept, n_features);
  } else {
    solve_dense(use_intercept, n_features);
  }
}

void SAGA::compute_step_corrections() {
  ulong n_features = model->get_n_features();
  std::shared_ptr<ModelLabelsFeatures> casted_model;
  casted_model = std::dynamic_pointer_cast<ModelLabelsFeatures>(model);
  ArrayDouble columns_sparsity = casted_model->get_column_sparsity_view();
  steps_correction = ArrayDouble(n_features);
  for (ulong j = 0; j < n_features; ++j) {
    steps_correction[j] = 1. / columns_sparsity[j];
  }
  ready_step_corrections = true;
}

void SAGA::solve_dense(bool use_intercept, ulong n_features) {
  double n_samples = model->get_n_samples();
  for (ulong t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Get the features matrix. We know that it's dense
    BaseArrayDouble x_i = model->get_features(i);
    double grad_i_factor = model->grad_i_factor(i, iterate);
    double grad_i_factor_old = gradients_memory[i];
    // Update gradient memory
    gradients_memory[i] = grad_i_factor;
    double grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong j = 0; j < n_features; ++j) {
      double x_ij = x_i._value_dense(j);
      double grad_avg_j = gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_ij + grad_avg_j);
      // Update the gradients average over seen samples
      gradients_average[j] += grad_factor_diff * x_ij / n_samples;
    }
    // deal with intercept here
    if (use_intercept) {
      iterate[n_features] -= step * (grad_factor_diff + gradients_average[n_features]);
      gradients_average[n_features] += grad_factor_diff / n_samples;
    }
    // Call the prox on the iterate
    prox->call(iterate, step, iterate);
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
  t += epoch_size;
}

void SAGA::solve_sparse_proba_updates(bool use_intercept, ulong n_features) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the current
  // support (non-zero values) of the sampled vector of features
  std::shared_ptr<ProxSeparable> casted_prox;
  if (prox->is_separable()) {
    casted_prox = std::static_pointer_cast<ProxSeparable>(prox);
  } else {
    TICK_ERROR("SAGA::solve_sparse_proba_updates can be used with a separable prox only.")
  }
  double n_samples = model->get_n_samples();
  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);
    double grad_i_factor = model->grad_i_factor(i, iterate);
    double grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    double grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      double x_ij = x_i.data()[idx_nnz];
      double grad_avg_j = gradients_average[j];
      // Step-size correction for coordinate j
      double step_correction = steps_correction[j];
      iterate[j] -= step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);
      gradients_average[j] += grad_factor_diff * x_ij / n_samples;
      // Prox is separable, apply regularization on the current coordinate
      casted_prox->call_single(j, iterate, step * step_correction, iterate);
    }
    // And let's not forget to update the intercept as well. It's updated at each step, so no step-correction.
    // Note that we call the prox, in order to be consistent with the dense case (in the case where the user
    // has the weird desire to to regularize the intercept)
    if (use_intercept) {
      iterate[n_features] -= step * (grad_factor_diff + gradients_average[n_features]);
      gradients_average[n_features] += grad_factor_diff / n_samples;
      casted_prox->call_single(n_features, iterate, step, iterate);
    }
    // Note that the average option for variance reduction with sparse data is a very bad idea,
    // but this is caught in the python class
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  t += epoch_size;
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
}

void SAGA::set_starting_iterate(ArrayDouble &new_iterate) {
  StoSolver::set_starting_iterate(new_iterate);
  next_iterate = iterate;
}
