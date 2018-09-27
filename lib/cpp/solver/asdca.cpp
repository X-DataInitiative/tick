// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/asdca.h"
//#include "tick/linear_model/model_poisreg.h"

template <class T>
AtomicSDCA<T>::AtomicSDCA(T l_l2sq, ulong epoch_size, T tol, RandType rand_type,
                   int record_every, int seed, int n_threads)
    : TBaseSDCA<T, std::atomic<T>>(l_l2sq, epoch_size, tol, rand_type, record_every, seed,
                                   n_threads) {
}

template <class T>
void AtomicSDCA<T>::update_delta_dual_i(const ulong i, const double delta_dual_i,
                                   const BaseArray<T> &feature_i, const double _1_over_lbda_n) {

  T dual_i = dual_vector[i].load();
  while (!dual_vector[i].compare_exchange_weak(dual_i,
                                               dual_i + delta_dual_i)) {
  }

  // Keep the last ascent seen for warm-starting sdca_dual_min_i
  delta[i] = delta_dual_i;

  // TODO: if model is sparse
  for (ulong idx_nnz = 0; idx_nnz < feature_i.size_sparse(); ++idx_nnz) {
    // Get the index of the idx-th sparse feature of feature_i
    ulong j = feature_i.indices()[idx_nnz];
    T feature_ij = feature_i.data()[idx_nnz];

    T tmp_iterate_j = tmp_primal_vector[j].load();
    while (!tmp_primal_vector[j].compare_exchange_weak(
        tmp_iterate_j,
        tmp_iterate_j + (delta_dual_i * feature_ij * _1_over_lbda_n))) {
    }
  }
  if (model->use_intercept()) {
    T tmp_iterate_j = tmp_primal_vector[model->get_n_features()];
    while (!tmp_primal_vector[model->get_n_features()].compare_exchange_weak(
        tmp_iterate_j,
        tmp_iterate_j + (delta_dual_i * _1_over_lbda_n))) {
    }
  }
}



template class DLL_PUBLIC AtomicSDCA<double>;
template class DLL_PUBLIC AtomicSDCA<float>;
