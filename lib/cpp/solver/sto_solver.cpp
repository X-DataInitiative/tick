// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/sto_solver.h"

template <class T, class K>
void TStoSolver<T, K>::init_permutation() {
  if ((rand_type == RandType::perm) && (rand_max > 0)) {
    permutation = ArrayULong(rand_max);
    for (ulong i = 0; i < rand_max; ++i) permutation[i] = i;
  }
}

template <class T, class K>
void TStoSolver<T, K>::reset() {
  t = 1;
  if (rand_type == RandType::perm) {
    i_perm = 0;
    shuffle();
  }
}

template <class T, class K>
ulong TStoSolver<T, K>::get_next_i() {
  ulong i = 0;
  if (rand_type == RandType::unif) {
    i = rand_unif(rand_max - 1);
  } else if (rand_type == RandType::perm) {
    if (!permutation_ready) {
      shuffle();
    }
    i = permutation[i_perm];
    i_perm++;
    if (i_perm >= rand_max) {
      shuffle();
    }
  }
  return i;
}

// Simulation of a random permutation using Knuth's algorithm
template <class T, class K>
void TStoSolver<T, K>::shuffle() {
  if (rand_type == RandType::perm) {
    // A secure check
    if (permutation.size() != rand_max) {
      init_permutation();
    }
    // Restart the i_perm
    i_perm = 0;

    for (ulong i = 1; i < rand_max; ++i) {
      // uniform number in { 0, ..., i }
      ulong j = rand_unif(i);
      // Exchange permutation[i] and permutation[j]
      ulong tmp = permutation[i];
      permutation[i] = permutation[j];
      permutation[j] = tmp;
    }
  }
  permutation_ready = true;
}

template <class T, class K>
void TStoSolver<T, K>::get_minimizer(Array<T> &out) {
  for (ulong i = 0; i < iterate.size(); ++i) {
    out[i] = iterate[i];
  }
}

template <class T, class K>
void TStoSolver<T, K>::get_iterate(Array<T> &out) {
  for (ulong i = 0; i < iterate.size(); ++i) out[i] = iterate[i];
}

template <class T, class K>
void TStoSolver<T, K>::set_starting_iterate(Array<K> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i] = new_iterate[i];
  }
}

template <>
void TStoSolver<double, std::atomic<double>>::set_starting_iterate(
    Array<std::atomic<double>> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i].store(new_iterate[i].load());
  }
}
template <>
void TStoSolver<float, std::atomic<float>>::set_starting_iterate(
    Array<std::atomic<float>> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i].store(new_iterate[i].load());
  }
}

template class TStoSolver<double, double>;
template class TStoSolver<float, float>;

template class TStoSolver<double, std::atomic<double>>;
template class TStoSolver<float, std::atomic<float>>;
