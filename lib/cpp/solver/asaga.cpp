// // License: BSD 3 clause

// #include "tick/solver/asaga.h"

// template <class T>
// AtomicSAGA<T>::AtomicSAGA(ulong epoch_size, T tol,
//                           RandType rand_type, T step, size_t record_every, int seed, int n_threads)
//     : TBaseSAGA<T, T>(epoch_size, tol, rand_type, step, record_every, seed),
//       n_threads(n_threads) {
//   un_threads = (size_t)n_threads;
// }

// template <class T>
// void AtomicSAGA<T>::initialize_solver() {
//   ulong n_samples = model->get_n_samples();
//   gradients_memory = Array<std::atomic<T>>(n_samples);
//   gradients_average = Array<std::atomic<T>>(model->get_n_coeffs());
//   gradients_memory.fill(0);
//   gradients_average.fill(0);
//   solver_ready = true;
// }

// template <class T>
// void AtomicSAGA<T>::threaded_solve(size_t n_epochs, size_t n_thread) {
//   T n_samples = model->get_n_samples();
//   T n_samples_inverse = 1 / n_samples;

//   bool use_intercept = model->use_intercept();
//   ulong n_features = model->get_n_features();

//   T x_ij = 0;
//   T step_correction = 0;
//   T grad_factor_diff = 0, grad_avg_j = 0;
//   T grad_i_factor = 0, grad_i_factor_old = 0;

//   ulong idx_nnz = 0;
//   ulong thread_epoch_size = epoch_size / n_threads;
//   thread_epoch_size += n_thread < (epoch_size % n_threads);

//   auto start = std::chrono::steady_clock::now();

//   for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
//     for (ulong t = 0; t < thread_epoch_size; ++t) {
//       // Get next sample index
//       ulong i = get_next_i();
//       // Sparse features vector
//       BaseArray<T> x_i = model->get_features(i);
//       grad_i_factor = model->grad_i_factor(i, iterate);
//       grad_i_factor_old = gradients_memory[i].load();

//       while (!gradients_memory[i].compare_exchange_weak(grad_i_factor_old,
//                                                         grad_i_factor)) {
//       }

//       grad_factor_diff = grad_i_factor - grad_i_factor_old;
//       for (idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
//         // Get the index of the idx-th sparse feature of x_i
//         ulong j = x_i.indices()[idx_nnz];
//         x_ij = x_i.data()[idx_nnz];
//         grad_avg_j = gradients_average[j].load();
//         // Step-size correction for coordinate j
//         step_correction = steps_correction[j];

//         while (!gradients_average[j].compare_exchange_weak(
//             grad_avg_j,
//             grad_avg_j + (grad_factor_diff * x_ij * n_samples_inverse))) {
//         }

//         // Prox is separable, apply regularization on the current coordinate
//         iterate[j] = casted_prox->call_single(
//             iterate[j] - (step * (grad_factor_diff * x_ij +
//                 step_correction * grad_avg_j)),
//             step * step_correction);
//       }
//       // And let's not forget to update the intercept as well. It's updated at
//       // each step, so no step-correction. Note that we call the prox, in order
//       // to be consistent with the dense case (in the case where the user has
//       // the weird desire to to regularize the intercept)
//       if (use_intercept) {
//         iterate[n_features] -=
//             step * (grad_factor_diff + gradients_average[n_features]);
//         T gradients_average_j = gradients_average[n_features];
//         while (!gradients_average[n_features].compare_exchange_weak(
//             gradients_average_j,
//             gradients_average_j + (grad_factor_diff / n_samples))) {
//         }
//         casted_prox->call_single(n_features, iterate, step, iterate);
//       }
//     }

//     // Record only on one thread
//     if (n_thread == 0) {
//       TBaseSAGA<T, T>::t += epoch_size;
//       if ((last_record_epoch + epoch) == 1 || ((last_record_epoch + epoch) % record_every == 0)) {
//         auto end = std::chrono::steady_clock::now();
//         double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
//             static_cast<double>(std::chrono::steady_clock::period::den);
//         save_history(last_record_time + time, last_record_epoch + epoch);
//       }
//     }
//   }

//   if (n_thread == 0) {
//     auto end = std::chrono::steady_clock::now();
//     double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
//         static_cast<double>(std::chrono::steady_clock::period::den);
//     last_record_time = time;
//     last_record_epoch += n_epochs;
//   }
// }

// template <class T>
// void AtomicSAGA<T>::solve(size_t n_epochs) {
//   // Data is sparse, and we use the probabilistic update strategy
//   // This means that the model is a child of ModelGeneralizedLinear.
//   // The strategy used here uses non-delayed updates, with corrected
//   // step-sizes using a probabilistic approximation and the
//   // penalization trick: with such a model and prox, we can work only inside the
//   // current support (non-zero values) of the sampled vector of features

//   prepare_solve();
//   if (prox->is_separable()) {
//     casted_prox = std::static_pointer_cast<TProxSeparable<T, T> >(prox);
//   } else {
//     TICK_ERROR(
//         "AtomicSAGA can be used with a separable prox only.")
//   }
//   if (!model->is_sparse()) {
//     TICK_ERROR("AtomicSAGA can be used with sparse features only")
//   }

//   std::vector<std::thread> threads;
//   for (size_t i = 0; i < un_threads; i++) {
//     threads.emplace_back(&AtomicSAGA<T>::threaded_solve, this, n_epochs, i);
//   }
//   for (size_t i = 0; i < un_threads; i++) {
//     threads[i].join();
//   }
// }

// template class DLL_PUBLIC AtomicSAGA<double>;
// template class DLL_PUBLIC AtomicSAGA<float>;
