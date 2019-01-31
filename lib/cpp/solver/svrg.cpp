// License: BSD 3 clause

#include "tick/solver/svrg.h"
#include "tick/base_model/model_labels_features.h"

template <class T, class K>
Array<T> init_from_array(Array<K> array_k){
  Array<T> array_t(array_k.size());
  array_t.init_to_zero();
  array_t.mult_incr(array_k, 1);
  return array_t;
};

template <class T, class K>
TSVRG<T, K>::TSVRG(ulong epoch_size, T tol, RandType rand_type, T step,
                   int record_every, int seed, int n_threads,
                   SVRG_VarianceReductionMethod variance_reduction,
                   SVRG_StepType step_method)
    : TStoSolver<T, K>(epoch_size, tol, rand_type, record_every, seed),
      n_threads(n_threads),
      step(step),
      variance_reduction(variance_reduction),
      ready_step_corrections(false),
      step_type(step_method) {}

template <class T, class K>
void TSVRG<T, K>::set_model(std::shared_ptr<TModel<T, K>> model) {
  TStoSolver<T, K>::set_model(model);
  ready_step_corrections = false;
}

template <class T, class K>
void TSVRG<T, K>::prepare_solve() {
  Array<T> previous_iterate;
  Array<T> previous_full_gradient;
  if (step_type == SVRG_StepType::BarzilaiBorwein && t > 1) {
    previous_iterate = init_from_array<T, K>(fixed_w);
    previous_full_gradient = full_gradient;
  }
  // The point where we compute the full gradient for variance reduction is the
  // new iterate obtained at the previous epoch
  next_iterate = init_from_array<T, K>(iterate);
  fixed_w = iterate;
  // Allocation and computation of the full gradient
  full_gradient = Array<T>(iterate.size());
  model->grad(fixed_w, full_gradient);

  if (step_type == SVRG_StepType::BarzilaiBorwein && t > 1) {
    Array<T> iterate_diff = init_from_array<T, K>(iterate);
    iterate_diff.mult_incr(previous_iterate, -1);
    Array<T> full_gradient_diff = full_gradient;
    full_gradient_diff.mult_incr(previous_full_gradient, -1);
    step = 1. / epoch_size * iterate_diff.norm_sq() /
           iterate_diff.dot(full_gradient_diff);
  }

  if ((model->is_sparse()) && (prox->is_separable())) {
    if (!ready_step_corrections) {
      compute_step_corrections();
    }
  } else {
    grad_i = Array<T>(iterate.size());
    grad_i_fixed_w = Array<T>(iterate.size());
  }
  rand_index = 0;
  if (variance_reduction == SVRG_VarianceReductionMethod::Random ||
      variance_reduction == SVRG_VarianceReductionMethod::Average) {
    next_iterate.init_to_zero();
  }
  if (variance_reduction == SVRG_VarianceReductionMethod::Random) {
    rand_index = rand_unif(epoch_size);
  }
}

template <class T, class K>
void TSVRG<T, K>::solve_one_epoch() {
  prepare_solve();
  if ((model->is_sparse()) && (prox->is_separable())) {
    bool use_intercept = model->use_intercept();
    ulong n_features = model->get_n_features();
    solve_sparse_proba_updates(use_intercept, n_features);
  } else {
    solve_dense();
  }
}

template <class T, class K>
void TSVRG<T, K>::compute_step_corrections() {
  ulong n_features = model->get_n_features();
  std::shared_ptr<TModelLabelsFeatures<T, K>> casted_model;
  casted_model = std::dynamic_pointer_cast<TModelLabelsFeatures<T, K>>(model);
  Array<T> columns_sparsity = casted_model->get_column_sparsity_view();
  steps_correction = Array<T>(n_features);
  for (ulong j = 0; j < n_features; ++j) {
    steps_correction[j] = 1. / columns_sparsity[j];
  }
  ready_step_corrections = true;
}

template <class T, class K>
void TSVRG<T, K>::solve_dense() {
  if (n_threads > 1) {
    std::vector<std::thread> threadsV;
    for (int i = 0; i < n_threads; i++) {
      threadsV.emplace_back([=](int n_thread) mutable -> void {
        std::mt19937_64 gen = get_generator(n_thread);
        for (ulong t = 0; t < (epoch_size / n_threads); ++t) {
          ulong next_i(get_next_i(&gen));
          dense_single_thread_solver(next_i);
        }
      }, i);
    }
    for (int i = 0; i < n_threads; i++) {
      threadsV[i].join();
    }

  } else {
    for (ulong t = 0; t < epoch_size; ++t) {
      ulong next_i = get_next_i();
      dense_single_thread_solver(next_i);
    }
  }
  if (variance_reduction == SVRG_VarianceReductionMethod::Last) {
    next_iterate = init_from_array<T, K>(iterate);
  }
  TStoSolver<T, K>::t += epoch_size;
}

template <class T, class K>
void TSVRG<T, K>::solve_sparse_proba_updates(bool use_intercept,
                                             ulong n_features) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the
  // current support (non-zero values) of the sampled vector of features
  if (prox->is_separable()) {
    casted_prox = std::static_pointer_cast<TProxSeparable<T, K>>(prox);
  } else {
    TICK_ERROR(
        "TSVRG<T, K>::solve_sparse_proba_updates can be used with a separable "
        "prox only.")
  }
  if (n_threads > 1) {
    std::vector<std::thread> threadsV;
    for (int i = 0; i < n_threads; i++) {
      threadsV.emplace_back([=]() mutable -> void {
        for (ulong t = 0; t < (epoch_size / n_threads); ++t) {
          ulong next_i(get_next_i());
          sparse_single_thread_solver(next_i, n_features, use_intercept);
        }
      });
    }
    for (int i = 0; i < n_threads; i++) {
      threadsV[i].join();
    }
  } else {
    for (ulong t = 0; t < epoch_size; ++t) {
      ulong next_i = get_next_i();
      sparse_single_thread_solver(next_i, n_features, use_intercept);
    }
  }

  if (variance_reduction == SVRG_VarianceReductionMethod::Last) {
    next_iterate = init_from_array<T, K>(iterate);
  }
  TStoSolver<T, K>::t += epoch_size;
}

template <class T, class K>
void TSVRG<T, K>::set_starting_iterate(Array<T>& new_iterate) {
  TStoSolver<T, K>::set_starting_iterate(new_iterate);
  next_iterate = init_from_array<T, K>(iterate);
}

template <class T, class K>
void TSVRG<T, K>::dense_single_thread_solver(const ulong& next_i) {
  const ulong& i = next_i;
  model->grad_i(i, iterate, grad_i);
  model->grad_i(i, fixed_w, grad_i_fixed_w);
  for (ulong j = 0; j < iterate.size(); ++j) {
    iterate[j] =
        iterate[j] - step * (grad_i[j] - grad_i_fixed_w[j] + full_gradient[j]);
  }
  prox->call(iterate, step, iterate);
  if (variance_reduction == SVRG_VarianceReductionMethod::Random &&
      t == rand_index) {
    next_iterate = init_from_array<T, K>(iterate);
  }
  if (variance_reduction == SVRG_VarianceReductionMethod::Average) {
    next_iterate.mult_incr(iterate, 1.0 / epoch_size);
  }
}

template <class T, class K>
void TSVRG<T, K>::sparse_single_thread_solver(
    const ulong& next_i, const ulong& n_features, const bool use_intercept) {
  const ulong& i = next_i;
  // Sparse features vector
  BaseArray<T> x_i = model->get_features(i);
  // Gradients factors (model is a GLM)
  // TODO: a grad_i_factor(i, array1, array2) to loop once on the features
  T grad_i_diff =
      model->grad_i_factor(i, iterate) - model->grad_i_factor(i, fixed_w);
  // We update the iterate within the support of the features vector, with the
  // probabilistic correction
  for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
    // Get the index of the idx-th sparse feature of x_i
    ulong j = x_i.indices()[idx_nnz];
    T full_gradient_j = full_gradient[j];
    // Step-size correction for coordinate j
    T step_correction = steps_correction[j];
    // Gradient descent with probabilistic step-size correction
    update_iterate(j, x_i.data()[idx_nnz], grad_i_diff, step_correction, full_gradient_j);
  }
  // And let's not forget to update the intercept as well. It's updated at each
  // step, so no step-correction. Note that we call the prox, in order to be
  // consistent with the dense case (in the case where the user has the weird
  // desire to to regularize the intercept)
  if (use_intercept) {
    update_iterate(n_features, 1., grad_i_diff, 1., full_gradient[n_features]);
  }
  // Note that the average option for variance reduction with sparse data is a
  // very bad idea, but this is caught in the python class
  if (variance_reduction == SVRG_VarianceReductionMethod::Random &&
      t == rand_index) {
    next_iterate = init_from_array<T, K>(iterate);
  }
  if (variance_reduction == SVRG_VarianceReductionMethod::Average) {
    next_iterate.mult_incr(iterate, 1.0 / epoch_size);
  }
}

// if K == std::atomic<T> (Wild case)
template <class T, class K>
template <class T1, class K1>
typename std::enable_if<std::is_same<T1, K1>::value, void>::type
TSVRG<T, K>::update_iterate(ulong j, T x_ij, T grad_factor_diff, T step_correction, T full_gradient_j) {
  T delta_iterate = - step * (x_ij * grad_factor_diff + step_correction * full_gradient_j);
  iterate[j] = casted_prox->call_single_with_index(
      iterate[j] + delta_iterate, step * step_correction, j);
};

// if K == std::atomic<T> (Atomic case)
template <class T, class K>
template <class T1, class K1>
typename std::enable_if<std::is_same<std::atomic<T1>, K1>::value, void>::type
TSVRG<T, K>::update_iterate(ulong j, T x_ij, T grad_factor_diff, T step_correction, T full_gradient_j) {
  T delta_iterate = - step * (x_ij * grad_factor_diff + step_correction * full_gradient_j);
  T iterate_j = iterate[j].load();

  T prox_delta_iterate = casted_prox->call_single_with_index(
      iterate_j + delta_iterate, step * step_correction, j) - iterate_j;

  // Prox is separable, apply regularization on the current coordinate
  while (!iterate[j].compare_exchange_weak(
      iterate_j,
      iterate_j + prox_delta_iterate)) {
  }
};


template class DLL_PUBLIC TSVRG<double>;
template class DLL_PUBLIC TSVRG<float>;

template class DLL_PUBLIC TSVRG<double, std::atomic<double>>;