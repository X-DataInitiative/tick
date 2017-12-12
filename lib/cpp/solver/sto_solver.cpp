// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/solver/sto_solver.h"

#include "tick/prox/prox_zero.h"

StoSolver::StoSolver(int seed)
    : seed(seed) {
    set_seed(seed);
    permutation_ready = false;
}

StoSolver::StoSolver(ulong epoch_size,
                     double tol,
                     RandType rand_type,
                     int seed)
    : prox(std::make_shared<ProxZero>(0.0)),
      epoch_size(epoch_size),
      tol(tol),
      rand_type(rand_type) {
    set_seed(seed);
    permutation_ready = false;
}

void StoSolver::init_permutation() {
    if ((rand_type == RandType::perm) && (rand_max > 0)) {
        permutation = ArrayULong(rand_max);
        for (ulong i = 0; i < rand_max; ++i)
            permutation[i] = i;
    }
}

void StoSolver::reset() {
    t = 1;
    if (rand_type == RandType::perm) {
        i_perm = 0;
        shuffle();
    }
}

ulong StoSolver::get_next_i() {
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
void StoSolver::shuffle() {
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

void StoSolver::get_minimizer(ArrayDouble &out) {
    for (ulong i = 0; i < iterate.size(); ++i)
        out[i] = iterate[i];
}

void StoSolver::get_iterate(ArrayDouble &out) {
    for (ulong i = 0; i < iterate.size(); ++i)
        out[i] = iterate[i];
}

void StoSolver::set_starting_iterate(ArrayDouble &new_iterate) {
    for (ulong i = 0; i < new_iterate.size(); ++i)
        iterate[i] = new_iterate[i];
}
