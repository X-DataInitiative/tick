// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/sto_solver.h"

#include "tick/prox/prox_zero.h"

template <class T, class K>
TStoSolver<T, K>::TStoSolver(int seed)
    : seed(seed) {
    set_seed(seed);
    permutation_ready = false;
}

template <class T, class K>
TStoSolver<T, K>::TStoSolver(ulong epoch_size,
                     K tol,
                     RandType rand_type,
                     int seed)
    : epoch_size(epoch_size),
      tol(tol),
      prox(std::make_shared<TProxZero<T, K> >(0.0)),
      rand_type(rand_type) {
    set_seed(seed);
    permutation_ready = false;
}

StoSolver::StoSolver(int seed)
    : TStoSolver(seed) {
}

StoSolver::StoSolver(
  ulong epoch_size,
  double tol,
  RandType rand_type,
  int seed
) : TStoSolver(epoch_size, tol, rand_type, seed)
{}

template <class T, class K>
void
TStoSolver<T, K>::init_permutation() {
    if ((rand_type == RandType::perm) && (rand_max > 0)) {
        permutation = ArrayULong(rand_max);
        for (ulong i = 0; i < rand_max; ++i)
            permutation[i] = i;
    }
}

template <class T, class K>
void
TStoSolver<T, K>::reset() {
    t = 1;
    if (rand_type == RandType::perm) {
        i_perm = 0;
        shuffle();
    }
}

template <class T, class K>
ulong
TStoSolver<T, K>::get_next_i() {
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
void
TStoSolver<T, K>::shuffle() {
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
void
TStoSolver<T, K>::get_minimizer(Array<K> &out) {
  for (ulong i = 0; i < iterate.size(); ++i) {
    out[i] = iterate[i];
  }
}

template <class T, class K>
void
TStoSolver<T, K>::get_iterate(Array<K> &out) {
  for (ulong i = 0; i < iterate.size(); ++i)
    out[i] = iterate[i];
}

template <class T, class K>
void
TStoSolver<T, K>::set_starting_iterate(Array<K> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i] = new_iterate[i];
  }
}

template class TStoSolver<double, double>;
template class TStoSolver<float , float>;
