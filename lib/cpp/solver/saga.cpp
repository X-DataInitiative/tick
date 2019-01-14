// License: BSD 3 clause

#include "tick/solver/saga.h"
#include "tick/base_model/model_generalized_linear.h"
#include "tick/prox/prox_separable.h"

template <class T, class K, class L>
TBaseSAGA<T, K, L>::TBaseSAGA(ulong epoch_size, T tol, RandType rand_type, T step,
                              int record_every, int seed, int n_threads)
    : TStoSolver<T, K>(epoch_size, tol, rand_type, record_every, seed),
      solver_ready(false),
      ready_step_corrections(false),
      step(step),
      n_threads(n_threads >= 1 ? n_threads : std::thread::hardware_concurrency()){}

template <class T, class K, class L>
void TBaseSAGA<T, K, L>::set_model(std::shared_ptr<TModel<T, K> > model) {
  // model must be a child of ModelGeneralizedLinear
  casted_model =
      std::dynamic_pointer_cast<TModelGeneralizedLinear<T, K> >(model);
  if (casted_model == nullptr) {
    TICK_ERROR("SAGA accepts only childs of `ModelGeneralizedLinear`")
  }
  TStoSolver<T, K>::set_model(model);
  ready_step_corrections = false;
  solver_ready = false;
}


template <class T, class K, class L>
void TBaseSAGA<T, K, L>::initialize_solver() {
  ulong n_samples = model->get_n_samples();
  gradients_memory = Array<L>(n_samples);
  gradients_memory.fill(0);
  gradients_average = Array<L>(model->get_n_coeffs());
  gradients_average.fill(0);
  solver_ready = true;

  auto start = std::chrono::steady_clock::now();
  casted_model->compute_features_norm_sq();
  auto end = std::chrono::steady_clock::now();
  double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
      static_cast<double>(std::chrono::steady_clock::period::den);
  last_record_time += time;
}

template <class T, class K, class L>
void TBaseSAGA<T, K, L>::prepare_solve() {
  if (!solver_ready) {
    initialize_solver();
  }
  if (model->is_sparse()) {
    if (prox->is_separable()) {
      casted_prox = std::static_pointer_cast<TProxSeparable<T, K> >(prox);
      if (!ready_step_corrections) {
        compute_step_corrections();
      }
    } else {
      TICK_ERROR(
          "SAGA on sparse models can be used with a separable prox only.")
    }
  }
}

template <class T, class K, class L>
void TBaseSAGA<T, K, L>::compute_step_corrections() {
  ulong n_features = model->get_n_features();
  Array<T> columns_sparsity = casted_model->get_column_sparsity_view();
  steps_correction = Array<T>(n_features);
  for (ulong j = 0; j < n_features; ++j) {
    steps_correction[j] = 1. / columns_sparsity[j];
  }
  ready_step_corrections = true;
}


template <class T, class K, class L>
void TBaseSAGA<T, K, L>::solve(int n_epochs) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the
  // current support (non-zero values) of the sampled vector of features
  bool use_intercept = model->use_intercept();
  ulong n_features = model->get_n_features();

  prepare_solve();

  auto lambda = [&](uint16_t n_thread) {
    T x_ij = 0;

    ulong idx_nnz = 0;
    ulong thread_epoch_size = epoch_size / n_threads;
    thread_epoch_size += n_thread < (epoch_size % n_threads);

    auto start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch < (n_epochs + 1); ++epoch) {
      for (ulong t = 0; t < thread_epoch_size; ++t) {
        // Get next sample index
        ulong i = get_next_i();
        // Sparse features vector
        BaseArray<T> x_i = model->get_features(i);

        T grad_factor_diff = update_gradient_memory(i);

        if (!model->is_sparse()) {
          for (ulong j = 0; j < n_features; ++j) {
            x_ij = x_i._value_dense(j);

            update_iterate_and_gradient_average(j, x_ij, grad_factor_diff, 1);
          }
        }
        else {
          for (idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
            // Get the index of the idx-th sparse feature of x_i
            ulong j = x_i.indices()[idx_nnz];
            x_ij = x_i.data()[idx_nnz];

            update_iterate_and_gradient_average(j, x_ij, grad_factor_diff, steps_correction[j]);
          }
        }
        // And let's not forget to update the intercept as well. It's updated at
        // each step, so no step-correction. Note that we call the prox, in order
        // to be consistent with the dense case (in the case where the user has
        // the weird desire to to regularize the intercept)
        if (use_intercept) {
          update_iterate_and_gradient_average(n_features, 1., grad_factor_diff, 1.);
        }

        // If prox was not called during iterate update
        if (!model->is_sparse()) prox->call(iterate, step, iterate);
      }

      // Record only on one thread
      if (n_thread == 0) {
        TStoSolver<T, K>::t += epoch_size;

        if ((last_record_epoch + epoch) == 1 || ((last_record_epoch + epoch) % record_every == 0)) {
          auto end = std::chrono::steady_clock::now();
          double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
              static_cast<double>(std::chrono::steady_clock::period::den);
          save_history(last_record_time + time, last_record_epoch + epoch);
        }
      }
    }

    if (n_thread == 0) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
          static_cast<double>(std::chrono::steady_clock::period::den);
      last_record_time = time;
      last_record_epoch += n_epochs;
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < n_threads; i++) {
    threads.emplace_back(lambda, i);
  }
  for (size_t i = 0; i < n_threads; i++) {
    threads[i].join();
  }

//  {
//    Array<T> average(gradients_average.size());
//    average.init_to_zero();
//    for(ulong i = 0; i < model->get_n_samples(); ++i){
//      average.mult_incr(model->get_features(i), gradients_memory[i] / model->get_n_samples());
//    }
//
//    average.print();
//    gradients_average.print();
//
//    for (ulong j = 0; j < average.size(); j ++) average[j] -= gradients_average[j];
//
//    std::cout << "Mean norm sq diff average = " << average.norm_sq() / average.size() << std::endl;
//  }
}


template <class T, class K, class L>
T TBaseSAGA<T, K, L>::update_gradient_memory(ulong i){
  TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

template <class T, class K, class L>
void TBaseSAGA<T, K, L>::update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                                             T step_correction){
  TICK_CLASS_DOES_NOT_IMPLEMENT("");
}


template class DLL_PUBLIC TBaseSAGA<double, double, double>;
template class DLL_PUBLIC TBaseSAGA<float, float, float>;

template class DLL_PUBLIC TBaseSAGA<double, double, std::atomic<double> >;
template class DLL_PUBLIC TBaseSAGA<float, float, std::atomic<float> >;



template <class T, class K>
TSAGA<T, K>::TSAGA(ulong epoch_size, T tol, RandType rand_type, T step, int record_every, int seed, int n_threads)
    : TBaseSAGA<T, K, T>(epoch_size, tol, rand_type, step, record_every, seed, n_threads) {}


template <class T, class K>
T TSAGA<T, K>::update_gradient_memory(ulong i){
  T grad_i_factor = model->grad_i_factor(i, iterate);
  T grad_i_factor_old = gradients_memory[i];
  gradients_memory[i] = grad_i_factor;

  return grad_i_factor - grad_i_factor_old;
}

template <class T, class K>
void TSAGA<T, K>::update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                                      T step_correction){
  T grad_avg_j = gradients_average[j];

  T delta_grad_avg_j = grad_factor_diff * x_ij / model->get_n_samples();
  T delta_iterate = -step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);

  gradients_average[j] += delta_grad_avg_j;

  // Prox is separable, apply regularization on the current coordinate
  T iterate_j_before_prox = iterate[j] + delta_iterate;
  if (model->is_sparse()) iterate[j] = casted_prox->call_single(
        iterate_j_before_prox, step * step_correction);
  else iterate[j] = iterate_j_before_prox;
}

template class DLL_PUBLIC TSAGA<double, double>;
template class DLL_PUBLIC TSAGA<double, std::atomic<double> >;
template class DLL_PUBLIC TSAGA<float, float>;
template class DLL_PUBLIC TSAGA<float, std::atomic<float> >;


template <class T, class K>
AtomicSAGA<T, K>::AtomicSAGA(ulong epoch_size, T tol,
                             RandType rand_type, T step, int record_every, int seed, int n_threads)
    : TBaseSAGA<T, K, std::atomic<T>>(epoch_size, tol, rand_type, step, record_every, seed,
                                      n_threads) {}

template <class T, class K>
T AtomicSAGA<T, K>::update_gradient_memory(ulong i){
  T grad_i_factor = model->grad_i_factor(i, iterate);

  T grad_i_factor_old;
  if (load_before_atomic) grad_i_factor_old = gradients_memory[i].load(custom_memory_order);

  while (!gradients_memory[i].compare_exchange_weak(
      grad_i_factor_old, grad_i_factor,
      custom_memory_order, custom_memory_order)) {
  }

  return grad_i_factor - grad_i_factor_old;
}

template <class T, class K>
void AtomicSAGA<T, K>::update_iterate_and_gradient_average(ulong j, T x_ij, T grad_factor_diff,
                                                           T step_correction){
  T grad_avg_j;
  if (load_before_atomic) grad_avg_j = gradients_average[j].load(custom_memory_order);

  T delta_grad_avg_j = grad_factor_diff * x_ij / model->get_n_samples();
  T delta_iterate = -step * (grad_factor_diff * x_ij + step_correction * grad_avg_j);

  while (!gradients_average[j].compare_exchange_weak(
      grad_avg_j,
      grad_avg_j + delta_grad_avg_j,
      custom_memory_order, custom_memory_order)) {
  }

  // Prox is separable, apply regularization on the current coordinate
  iterate[j] = casted_prox->call_single(
      iterate[j] + delta_iterate, step * step_correction);
}

template class DLL_PUBLIC AtomicSAGA<double, double>;
template class DLL_PUBLIC AtomicSAGA<double, std::atomic<double> >;
template class DLL_PUBLIC AtomicSAGA<float, float>;
template class DLL_PUBLIC AtomicSAGA<float, std::atomic<float> >;