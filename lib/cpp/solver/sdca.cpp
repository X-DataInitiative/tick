// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/sdca.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/prox/prox_zero.h"

template <class T, class K>
TBaseSDCA<T, K>::TBaseSDCA(T l_l2sq, ulong epoch_size, T tol, RandType rand_type,
                   int record_every, int seed, int n_threads)
    : TStoSolver<T, K>(epoch_size, tol, rand_type, record_every, seed), l_l2sq(l_l2sq),
      n_threads(n_threads >= 1 ? n_threads : std::thread::hardware_concurrency()){
  stored_variables_ready = false;
}

template <class T>
TSDCA<T>::TSDCA(T l_l2sq, ulong epoch_size, T tol, RandType rand_type, int record_every, int seed)
    : TBaseSDCA<T, T>(l_l2sq, epoch_size, tol, rand_type, record_every, seed) {
}

template <class T, class K>
void TBaseSDCA<T, K>::set_model(std::shared_ptr<TModel<T, K> > model) {
  // model must be a child of ModelGeneralizedLinear
  casted_model = std::dynamic_pointer_cast<TModelGeneralizedLinear<T, K> >(model);
  if (casted_model == nullptr) {
    TICK_ERROR("SDCA accepts only childs of `ModelGeneralizedLinear`")
  }

  TStoSolver<T, K>::set_model(model);
  this->model = model;
  this->n_coeffs = model->get_n_coeffs();
  stored_variables_ready = false;
}

template <class T, class K>
void TBaseSDCA<T, K>::reset() {
  TStoSolver<T, K>::reset();
  set_starting_iterate();
}


template <class T, class K>
void TBaseSDCA<T, K>::prepare_solve() {
  if (!stored_variables_ready) {
    set_starting_iterate();
  }
  if (model->is_sparse()) {
    if (prox->is_separable()) {
      casted_prox = std::static_pointer_cast<TProxSeparable<T, K> >(prox);
    } else {
      TICK_ERROR(
          "SDCA on sparse models can be used with a separable prox only.")
    }
  }

  // Compute features norm_sq in advance to avoid having multiple threads launching it
  auto start = std::chrono::steady_clock::now();
  casted_model->compute_features_norm_sq();
  auto end = std::chrono::steady_clock::now();
  double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
      static_cast<double>(std::chrono::steady_clock::period::den);
  last_record_time += time;
};

template <class T, class K>
void TBaseSDCA<T, K>::solve(int n_epochs) {
  prepare_solve();

  const SArrayULongPtr feature_index_map = model->get_sdca_index_map();
  const T scaled_l_l2sq = get_scaled_l_l2sq();

  const T _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);

  auto lambda = [&](uint16_t n_thread) {
    T dual_i = 0;
    T feature_ij = 0;
    T tmp_iterate_j = 0;
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

        const BaseArray<T> feature_i = model->get_features(feature_index);
        if (!model->is_sparse()) {
          // Update the primal variable

          // Call prox on the primal variable
          prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
        } else {
          // Get iterate ready
          for (ulong idx_nnz = 0; idx_nnz < feature_i.size_sparse(); ++idx_nnz) {
            ulong j = feature_i.indices()[idx_nnz];
            // Prox is separable, apply regularization on the current coordinate
            casted_prox->call_single(j, tmp_primal_vector, 1 / scaled_l_l2sq, iterate);
          }
          // And let's not forget to update the intercept as well. It's updated at
          // each step, so no step-correction. Note that we call the prox, in order to
          // be consistent with the dense case (in the case where the user has the
          // weird desire to to regularize the intercept)
          if (model->use_intercept()) {
            casted_prox->call_single(model->get_n_features(),
                                     tmp_primal_vector,
                                     1 / scaled_l_l2sq,
                                     iterate);
          }
        }

        // Maximize the dual coordinate i
        const T delta_dual_i = model->sdca_dual_min_i(
            feature_index, dual_vector[i], iterate, delta[i], scaled_l_l2sq);
        // Update the dual variable

        update_delta_dual_i(i, delta_dual_i, feature_i, _1_over_lbda_n);
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

  // Don't spawn threads if this is sequential
  if (n_threads == 1) {
    lambda(0);
  } else {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n_threads; i++) {
      threads.emplace_back(lambda, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
      threads[i].join();
    }
  }
}


template <class T, class K>
void TBaseSDCA<T, K>::solve_batch(int n_epochs, ulong batch_size) {
  prepare_solve();

  const SArrayULongPtr feature_index_map = model->get_sdca_index_map();
  const T scaled_l_l2sq = get_scaled_l_l2sq();
  const T _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);

  auto lambda = [&](uint16_t n_thread) {
    T dual_i = 0;
    T feature_ij = 0;
    T iterate_j = 0;
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

        for (ulong i = 0; i < batch_size; ++i) {
          feature_indices[i] = feature_index_map != nullptr ?
                               (*feature_index_map)[indices[i]] : indices[i];
        }

        precompute_sdca_dual_min_weights(
            batch_size, scaled_l_l2sq, _1_over_lbda_n, feature_indices, g, p);
        for (ulong k = 0; k < batch_size; ++k)
          sdca_labels[k] = casted_model->get_label(feature_indices[k]);

        delta_duals = model->sdca_dual_min_many(
            batch_size, duals, scaled_l_l2sq, g, n_hess, p, n_grad, sdca_labels, new_duals,
            delta_duals_tmp, ipiv);

        for (ulong k = 0; k < batch_size; ++k) {
          const ulong i = indices[k];
          const ulong feature_index = feature_indices[k];
          const double delta_dual_i = delta_duals[k];
          BaseArray<T> feature_i = model->get_features(feature_index);

          update_delta_dual_i(i, delta_dual_i, feature_i, _1_over_lbda_n);
        }
      }

      // Record only on one thread
      if (n_thread == 0) {
        t += epoch_size;

        if ((last_record_epoch + epoch) == 1 || ((last_record_epoch + epoch) % record_every == 0)) {
          auto end = std::chrono::steady_clock::now();
          double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
              static_cast<double>(std::chrono::steady_clock::period::den);

          // update iterate
          prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
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

  // Don't spawn threads if this is sequential
  if (n_threads == 1) {
    lambda(0);
  } else {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n_threads; i++) {
      threads.emplace_back(lambda, i);
    }
    for (size_t i = 0; i < n_threads; i++) {
      threads[i].join();
    }
  }

  // Put its final value in iterate
  prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
}


// This is the part corresponding to the gradient and the hessian of the ridge regularization
// used in the Newton descent steps
// The gradient of this part writes at coordinate i
// 1/n (w^\top x_i + 1 / (lambda n) \sum_{j \in batch} \delta \alpha x_i^\top x_j )
// The hessian of this part writes at coorindate i,j
// 1 / (lambda n^2) x_i^\top x_j
// Hence we compute the weights
// * p_i = w^top x_i
// * g_ij = x_i^\top x_j / (lambda n)
// In order to avoid maintaining iterate, we compute it on the fly
template <class T, class K>
void TBaseSDCA<T, K>::precompute_sdca_dual_min_weights(
    ulong batch_size, double scaled_l_l2sq, double _1_over_lbda_n, const ArrayULong &feature_indices,
    Array2d<T> &g, Array<T> &p) {

  for (ulong i = 0; i < batch_size; ++i) {
    const BaseArray<T> feature_i = model->get_features(feature_indices[i]);

    for (ulong j = 0; j < batch_size; ++j) {
      const BaseArray<T> feature_j = model->get_features(feature_indices[j]);

      if (j < i) g(i, j) = g(j, i);
      else if (i == j)
        g(i, i) = casted_model->get_features_norm_sq_i(feature_indices[i]) * _1_over_lbda_n;
      else
        g(i, j) = feature_i.dot(feature_j) * _1_over_lbda_n;

      if (model->use_intercept()) g(i, j) += _1_over_lbda_n;
    }
  }

  if (!model->is_sparse()) {
    prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
    for (ulong i = 0; i < batch_size; ++i)
      p[i] = casted_model->get_inner_prod(feature_indices[i], iterate);

  } else {
    p.init_to_zero();
    for (ulong k = 0; k < batch_size; ++k) {
      BaseArray<T> feature_i = model->get_features(feature_indices[k]);
      for (ulong idx_nnz = 0; idx_nnz < feature_i.size_sparse(); ++idx_nnz) {
        ulong j = feature_i.indices()[idx_nnz];
        T feature_ij = feature_i.data()[idx_nnz];

        // Prox is separable, apply regularization on the current coordinate
        T iterate_j = casted_prox->call_single_with_index(
            tmp_primal_vector[j], 1 / scaled_l_l2sq, j);
        p[k] += feature_ij * iterate_j;
      }
      // And let's not forget to update the intercept as well.
      // Note that we call the prox, in order to be consistent with the dense case (in the case
      // where the user has the weird desire to to regularize the intercept)
      if (model->use_intercept()) {
        T iterate_j = casted_prox->call_single_with_index(
            tmp_primal_vector[model->get_n_features()], 1 / scaled_l_l2sq, model->get_n_features());
        p[k] += iterate_j;
      }
    }
  }
};

template <class T, class K>
void TBaseSDCA<T, K>::update_delta_dual_i(ulong i, double delta_dual_i,
                                          const BaseArray<T> &feature_i, double _1_over_lbda_n) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("");
};

template <class T>
void TSDCA<T>::update_delta_dual_i(ulong i, double delta_dual_i,
                                   const BaseArray<T> &feature_i, double _1_over_lbda_n) {
  delta[i] = delta_dual_i;
  dual_vector[i] += delta_dual_i;

  if (model->use_intercept()) {
    Array<T> primal_features = view(tmp_primal_vector, 0, feature_i.size());
    primal_features.mult_incr(feature_i, delta_dual_i * _1_over_lbda_n);
    tmp_primal_vector[model->get_n_features()] += delta_dual_i * _1_over_lbda_n;
  } else {
    tmp_primal_vector.mult_incr(feature_i, delta_dual_i * _1_over_lbda_n);
  }
}


template <class T, class K>
void TBaseSDCA<T, K>::set_starting_iterate() {
  if (dual_vector.size() != rand_max) dual_vector = Array<K>(rand_max);

  dual_vector.init_to_zero();

  // If it is not ModelPoisReg, primal vector will be full of 0 as dual vector
  bool can_initialize_primal_to_zero = true;
  if (dynamic_cast<ModelPoisReg *>(model.get())) {
    std::shared_ptr<ModelPoisReg> casted_model =
        std::dynamic_pointer_cast<ModelPoisReg>(model);
    if (casted_model->get_link_type() == LinkType::identity) {
      can_initialize_primal_to_zero = false;
    }
  }

  if (can_initialize_primal_to_zero) {
    if (tmp_primal_vector.size() != n_coeffs)
      tmp_primal_vector = Array<K>(n_coeffs);

    if (iterate.size() != n_coeffs) iterate = Array<K>(n_coeffs);

    if (delta.size() != rand_max) delta = Array<T>(rand_max);

    iterate.init_to_zero();
    delta.init_to_zero();
    tmp_primal_vector.init_to_zero();
    stored_variables_ready = true;
  } else {
    set_starting_iterate(dual_vector);
  }
}

template <class T, class K>
void TBaseSDCA<T, K>::set_starting_iterate(Array<K> &dual_vector) {
  if (dual_vector.size() != rand_max) {
    TICK_ERROR("Starting iterate should be dual vector and have shape ("
                   << rand_max << ", )");
  }

  if (!dynamic_cast<TProxZero<T, K> *>(prox.get())) {
    TICK_ERROR(
        "set_starting_iterate in SDCA might be call only if prox is ProxZero. "
        "Otherwise "
        "we need to implement the Fenchel conjugate of the prox gradient");
  }

  if (iterate.size() != n_coeffs) iterate = Array<K>(n_coeffs);
  if (delta.size() != rand_max) delta = Array<T>(rand_max);

  this->dual_vector = dual_vector;
  model->sdca_primal_dual_relation(get_scaled_l_l2sq(), dual_vector, iterate);
  tmp_primal_vector = iterate;

  stored_variables_ready = true;
}

template class DLL_PUBLIC TBaseSDCA<double, double>;
template class DLL_PUBLIC TBaseSDCA<float, float>;

template class DLL_PUBLIC TSDCA<double>;
template class DLL_PUBLIC TSDCA<float>;

template class DLL_PUBLIC TBaseSDCA<double, std::atomic<double> >;
template class DLL_PUBLIC TBaseSDCA<float, std::atomic<float> >;