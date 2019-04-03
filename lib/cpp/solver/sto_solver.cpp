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
  time_history.clear();
  iterate_history.clear();
  epoch_history.clear();
  last_record_epoch = 0;
  last_record_time = 0;
}

template <class T, class K>
ulong TStoSolver<T, K>::get_next_i(std::mt19937_64* gen) {
  ulong i = 0;
  if (rand_type == RandType::unif) {
    i = rand_unif(rand_max - 1, gen);
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
void TStoSolver<T, K>::solve(size_t n_epochs) {
  double initial_time = last_record_time;
  int initial_epoch = last_record_epoch;

  auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    Interruption::throw_if_raised();

    solve_one_epoch();

    if ((initial_epoch + epoch) == 1 || ((initial_epoch + epoch) % record_every == 0)) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
          static_cast<double>(std::chrono::steady_clock::period::den);
      save_history(initial_time + time, initial_epoch + epoch);
    }
  }

  auto end = std::chrono::steady_clock::now();
  double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
      static_cast<double>(std::chrono::steady_clock::period::den);
  last_record_time = time;
  last_record_epoch = initial_epoch + n_epochs;
}

template <class T, class K>
void TStoSolver<T, K>::save_history(double time, int epoch) {
  time_history.emplace_back(time);
  epoch_history.emplace_back(epoch);

  iterate_history.emplace_back(iterate.size());
  get_iterate(iterate_history.back());
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
std::vector<std::shared_ptr<SArray<T> > >  TStoSolver<T, K>:: get_iterate_history() const {
  std::vector<std::shared_ptr<SArray<T> > > shared_iterate_history(0);
  for (ulong i = 0; i < iterate_history.size(); ++i) {
    Array<T> copied_iterate_i = iterate_history[i];
    shared_iterate_history.push_back(copied_iterate_i.as_sarray_ptr());
  }
  return shared_iterate_history;
}

template <class T, class K>
void TStoSolver<T, K>::set_starting_iterate(Array<T> &new_iterate) {
  for (ulong i = 0; i < new_iterate.size(); ++i) {
    iterate[i] = new_iterate[i];
  }
}

template class TStoSolver<double, double>;
template class TStoSolver<float, float>;

template class TStoSolver<double, std::atomic<double>>;
template class TStoSolver<float, std::atomic<float>>;
