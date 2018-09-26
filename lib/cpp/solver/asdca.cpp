// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/asdca.h"
//#include "tick/linear_model/model_poisreg.h"

template <class T>
AtomicSDCA<T>::AtomicSDCA(T l_l2sq, ulong epoch_size, T tol, RandType rand_type,
                   int record_every, int seed, int n_threads)
    : TBaseSDCA<T, std::atomic<T>>(l_l2sq, epoch_size, tol, rand_type, record_every, seed),
      n_threads(n_threads) {
  un_threads = (size_t)n_threads;
}

template <class T>
void AtomicSDCA<T>::solve(int n_epochs) {
  if (!stored_variables_ready) {
    set_starting_iterate();
  }
  if (!ready_step_corrections) compute_step_corrections();

  const SArrayULongPtr feature_index_map = model->get_sdca_index_map();
  const T scaled_l_l2sq = get_scaled_l_l2sq();

  const T _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);

  auto lambda = [&](uint16_t n_thread) {
    T dual_i = 0;
    T x_ij = 0;
    T iterate_j = 0;
    ulong n_features = model->get_n_features();

    ulong idx_nnz = 0;
    ulong thread_epoch_size = epoch_size / n_threads;
    thread_epoch_size += n_thread < (epoch_size % n_threads);

    auto start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch < (n_epochs + 1); ++epoch) {
      for (ulong t = 0; t < thread_epoch_size; ++t) {
        // Pick i uniformly at random
        ulong i = get_next_i();
        ulong feature_index = i;
        if (feature_index_map != nullptr) {
          feature_index = (*feature_index_map)[i];
        }

        // Maximize the dual coordinate i
        const T delta_dual_i = model->sdca_dual_min_i(
            feature_index, dual_vector[i], iterate, delta[i], scaled_l_l2sq);
        // Update the dual variable

        dual_i = dual_vector[i].load();
        while (!dual_vector[i].compare_exchange_weak(dual_i,
                                                     dual_i + delta_dual_i)) {
        }

        // Keep the last ascent seen for warm-starting sdca_dual_min_i
        delta[i] = delta_dual_i;

        BaseArray<T> x_i = model->get_features(feature_index);
        for (idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
          // Get the index of the idx-th sparse feature of x_i
          ulong j = x_i.indices()[idx_nnz];
          x_ij = x_i.data()[idx_nnz];

          iterate_j = iterate[j].load();
          while (!iterate[j].compare_exchange_weak(
              iterate_j,
              iterate_j + (delta_dual_i * x_ij * _1_over_lbda_n))) {
          }
        }
        if (model->use_intercept()) {
          iterate_j = iterate[n_features];
          while (!iterate[n_features].compare_exchange_weak(
              iterate_j,
              iterate_j + (delta_dual_i * _1_over_lbda_n))) {
          }
        }
      }

      // Record only on one thread
      if (n_thread == 0) {
        t += epoch_size;

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
  for (size_t i = 0; i < un_threads; i++) {
    threads.emplace_back(lambda, i);
  }
  for (size_t i = 0; i < un_threads; i++) {
    threads[i].join();
  }
}


template <class T>
void AtomicSDCA<T>::solve_batch(int n_epochs, ulong batch_size) {
  if (!stored_variables_ready) {
    set_starting_iterate();
  }

  // TODO
//  auto casted_model = std::static_pointer_cast<TModelGeneralizedLinear<T, std::atomic<T>> >(model);
//  casted_model->compute_features_norm_sq();
  model->get_lip_max();
  if (!ready_step_corrections) compute_step_corrections();

  const SArrayULongPtr feature_index_map = model->get_sdca_index_map();
  const T scaled_l_l2sq = get_scaled_l_l2sq();

  const T _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);

  auto lambda = [&](uint16_t n_thread) {
    T dual_i = 0;
    T x_ij = 0;
    T iterate_j = 0;
    ulong n_features = model->get_n_features();
    Array<T> delta_duals(batch_size);
    Array<T> duals(batch_size);
    ArrayULong feature_indices(batch_size);
    ArrayULong indices(batch_size);

    Array<T> p = Array<T>(batch_size);
    Array2d<T> g = Array2d<T>(batch_size, batch_size);
    Array<T> sdca_labels = Array<T>(batch_size);

    Array<T> n_grad = Array<T>(batch_size);
    Array2d<T> n_hess = Array2d<T>(batch_size, batch_size);

    Array<T> new_duals = Array<T>(batch_size);
    Array<T> delta_duals_tmp = Array<T>(batch_size);

    ArrayInt ipiv = ArrayInt(batch_size);

    ulong idx_nnz = 0;
    ulong thread_epoch_size = epoch_size / n_threads;
    thread_epoch_size += n_thread < (epoch_size % n_threads);

    auto start = std::chrono::steady_clock::now();

    for (int epoch = 1; epoch < (n_epochs + 1); ++epoch) {
      for (ulong t = 0; t < thread_epoch_size; t += batch_size) {
        // Pick i uniformly at random
        indices.fill(rand_max + 1);
        for (ulong i = 0; i < batch_size; ++i) {
          ulong try_i = get_next_i();
          while (indices.contains(try_i)) try_i = get_next_i();
          indices[i] = try_i;
          duals[i] = dual_vector[indices[i]];
        }

        for (ulong i = 0; i < batch_size; ++i)
          feature_indices[i] = feature_index_map != nullptr?
              (*feature_index_map)[indices[i]]: indices[i];

        delta_duals = model->sdca_dual_min_many(indices, duals, iterate, scaled_l_l2sq, g, n_hess,
            p, n_grad, sdca_labels, new_duals, delta_duals, ipiv);


        for (ulong k = 0; k < batch_size; ++k) {
          const ulong i = indices[k];
          const ulong feature_index = feature_indices[k];
          const double delta_dual_i = delta_duals[k];

          dual_i = dual_vector[i].load();
          while (!dual_vector[i].compare_exchange_weak(dual_i,
                                                       dual_i + delta_dual_i)) {
          }
          // Keep the last ascent seen for warm-starting sdca_dual_min_i
          delta[i] = delta_dual_i;

          BaseArray<T> x_i = model->get_features(feature_index);
          for (idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
            // Get the index of the idx-th sparse feature of x_i
            ulong j = x_i.indices()[idx_nnz];
            x_ij = x_i.data()[idx_nnz];

            iterate_j = iterate[j].load();
            while (!iterate[j].compare_exchange_weak(
                iterate_j,
                iterate_j + (delta_dual_i * x_ij * _1_over_lbda_n))) {
            }
          }
          if (model->use_intercept()) {
            iterate_j = iterate[n_features];
            while (!iterate[n_features].compare_exchange_weak(
                iterate_j,
                iterate_j + (delta_dual_i * _1_over_lbda_n))) {
            }
          }
        }
      }

      // Record only on one thread
      if (n_thread == 0) {
        t += epoch_size;

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
  for (size_t i = 0; i < un_threads; i++) {
    threads.emplace_back(lambda, i);
  }
  for (size_t i = 0; i < un_threads; i++) {
    threads[i].join();
  }
}

//
//template <class T>
//void AtomicSDCA<T>::set_starting_iterate() {
//  if (dual_vector.size() != rand_max) dual_vector = Array<std::atomic<T>>(rand_max);
//
//  dual_vector.init_to_zero();
//
//  // If it is not ModelPoisReg, primal vector will be full of 0 as dual vector
//  bool can_initialize_primal_to_zero = true;
//  if (dynamic_cast<ModelPoisReg *>(model.get())) {
//    std::shared_ptr<ModelPoisReg> casted_model =
//        std::dynamic_pointer_cast<ModelPoisReg>(model);
//    if (casted_model->get_link_type() == LinkType::identity) {
//      can_initialize_primal_to_zero = false;
//    }
//  }
//
//  if (can_initialize_primal_to_zero) {
////    if (tmp_primal_vector.size() != n_coeffs)
////      tmp_primal_vector = Array<T>(n_coeffs);
//
//    if (iterate.size() != n_coeffs) iterate = Array<std::atomic<T>>(n_coeffs);
//
//    if (delta.size() != rand_max) delta = Array<T>(rand_max);
//
//    iterate.init_to_zero();
//    delta.init_to_zero();
////    tmp_primal_vector.init_to_zero();
//    stored_variables_ready = true;
//  } else {
//    set_starting_iterate(dual_vector);
//  }
//}
//
//template <class T>
//void AtomicSDCA<T>::set_starting_iterate(Array<std::atomic<T>> &dual_vector) {
//  if (dual_vector.size() != rand_max) {
//    TICK_ERROR("Starting iterate should be dual vector and have shape ("
//               << rand_max << ", )");
//  }
//
//  if (!dynamic_cast<TProxZero<T, std::atomic<T>> *>(prox.get())) {
//    TICK_ERROR(
//        "set_starting_iterate in SDCA might be call only if prox is ProxZero. "
//        "Otherwise "
//        "we need to implement the Fenchel conjugate of the prox gradient");
//  }
//
//  if (iterate.size() != n_coeffs) iterate = Array<std::atomic<T>>(n_coeffs);
//  if (delta.size() != rand_max) delta = Array<T>(rand_max);
//
//  this->dual_vector = dual_vector;
//  model->sdca_primal_dual_relation(get_scaled_l_l2sq(), dual_vector, iterate);
////  tmp_primal_vector = iterate;
//
//  stored_variables_ready = true;
//}

template class DLL_PUBLIC AtomicSDCA<double>;
template class DLL_PUBLIC AtomicSDCA<float>;
