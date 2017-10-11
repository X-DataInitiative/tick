// License: BSD 3 clause

#include "online_forest.h"

OnlineForest::OnlineForest(uint32_t n_trees, uint32_t n_min_samples,
                           uint8_t n_splits, CycleType cycle_type)
    : n_trees(n_trees), n_min_samples(n_min_samples),
      n_splits(n_splits), cycle_type(cycle_type) {
  has_data = false;
  // No iteration so far
  t = 0;
  //
  permutation_ready = false;
  // rand = Rand(123);
  i_perm = 0;
  for (auto i = 0; i < n_trees; ++i) {
    // ICICICI
    trees.emplace_back(this);
  }
}

// Do n_iter iterations
void OnlineForest::fit(ulong n_iter) {
  if (!has_data) {
    TICK_ERROR("OnlineForest::fit: the forest has no data yet.")
  }
  // Could be parallelized
  for (ulong it = 0; it < n_iter; ++it) {
    std::cout << "------------------" << std::endl;
    std::cout << "iteration=" << it << std::endl;
    ulong sample_index = get_next_sample();
    std::cout << "sample_index=" << sample_index << std::endl;
    for (auto &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(sample_index);
    }
    t++;
  }
}

void OnlineForest::init_permutation() {}

//// Simulation of a random permutation using Knuth's algorithm
//void OnlineForest::shuffle() {
//  if (cycle_type == CycleType::permutation) {
//    // A secure check
//    if (permutation.size() != get_n_samples()) {
//      init_permutation();
//    }
//    // Restart the i_perm
//    i_perm = 0;
//    for (ulong i = 1; i < get_n_samples(); ++i) {
//      // uniform number in { 0, ..., i }
//      ulong j = rand_unif(i);
//      // Exchange permutation[i] and permutation[j]
//      ulong tmp = permutation[i];
//      permutation[i] = permutation[j];
//      permutation[j] = tmp;
//    }
//  }
//  permutation_ready = true;
//}

ulong OnlineForest::get_next_sample() {
  ulong i = 0;
  if (cycle_type == CycleType::uniform) {
    // i = rand_unif(get_n_samples() - 1);
    i = 0;
  } else {
    if (cycle_type == CycleType::permutation) {
      if (!permutation_ready) {
        shuffle();
      }
      i = permutation[i_perm];
      i_perm++;
      if (i_perm >= get_n_samples()) {
        shuffle();
      }
    } else {
      // Otherwise it's cycling through the data
      i = i_perm;
      i_perm++;
      if (i_perm >= get_n_samples()) {
        i_perm = 0;
      }
    }
  }
  return i;
}

void OnlineForest::set_data(const SBaseArrayDouble2dPtr features,
                            const SArrayDoublePtr labels) {
  this->features = features;
  this->labels = labels;
  has_data = true;
}
