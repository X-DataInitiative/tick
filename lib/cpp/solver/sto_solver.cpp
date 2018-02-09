// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/sto_solver.h"

#include "tick/prox/prox_zero.h"

template <class T>
TStoSolver<T>::TStoSolver(int seed) : seed(seed) {
  set_seed(seed);
  permutation_ready = false;
}

template <class T>
TStoSolver<T>::TStoSolver(ulong epoch_size, T tol, RandType rand_type, int seed)
    : epoch_size(epoch_size),
      tol(tol),
      prox(std::make_shared<TProxZero<T> >(0.0)),
      rand_type(rand_type) {
  set_seed(seed);
  permutation_ready = false;
}

template <class T>
void TStoSolver<T>::init_permutation() {
  if ((rand_type == RandType::perm) && (rand_max > 0)) {
    permutation = ArrayULong(rand_max);
    for (ulong i = 0; i < rand_max; ++i) permutation[i] = i;
  }
}

template <class T>
void TStoSolver<T>::reset() {
  t = 1;
  if (rand_type == RandType::perm) {
    i_perm = 0;
    shuffle();
  }
}

template <class T>
ulong TStoSolver<T>::get_next_i() {
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
template <class T>
void TStoSolver<T>::shuffle() {
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

template <class T>
void TStoSolver<T>::get_minimizer(Array<T> &out) {
  for (ulong i = 0; i < iterate.size(); ++i) {
    out[i] = iterate[i];
  }
}

template <class T>
void TStoSolver<T>::get_iterate(Array<T> &out) {
  for (ulong i = 0; i < iterate.size(); ++i) out[i] = iterate[i];
}

template <class T>
void TStoSolver<T>::set_starting_iterate(Array<T> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i] = new_iterate[i];
  }
}

template class TStoSolver<double>;
template class TStoSolver<float>;
