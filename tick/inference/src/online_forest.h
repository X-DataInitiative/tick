#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include "node.h"
#include "tree.h"

// Type of randomness used when sampling at random data points
enum class CycleType {
  uniform = 0,
  permutation,
  sequential
};

class OnlineForest {
 private:
  // Number of Trees in the forest
  uint32_t n_trees;
  // Number of samples required in a node before splitting it (this means that we wait to have
  // n_min_samples in the node before computing the candidate splits and impurities, which uses
  // the range of the samples)
  uint32_t n_min_samples;
  // Number of candidate splits to be considered
  uint8_t n_splits;
  // Iteration counter
  ulong t;
  // The vector of features
  SBaseArrayDouble2dPtr features;
  // The vector of labels
  SArrayDoublePtr labels;
  // The list of trees in the forest
  std::vector<Tree> trees;
  // Do the forest received data
  bool has_data;

  // Type of random sampling
  CycleType cycle_type;

  // An array that allows to store the sampled random permutation
  ArrayULong permutation;

  // Current index in the permutation (useful when using random permutation sampling)
  ulong i_perm;

  // A flag that specify if random permutation is ready to be used or not
  bool permutation_ready;

  // Init permutation array in case of Random is srt to permutation
  void init_permutation();

  // Rand rand;

  ulong get_next_sample();

  void shuffle();

  // Returns a uniform integer in the set {0, ..., m - 1}
//  inline ulong rand_unif(ulong m) {
//    return rand.uniform_int(ulong{0}, m);
//  }


 public:
  OnlineForest(uint32_t n_trees, uint32_t n_min_samples, uint8_t n_splits, CycleType cycle_type=CycleType::sequential);

  // Fit the forest by doing a certain number number of iterations
  void fit(ulong n_iter = 0);

  // Pass the data to the forest
  void set_data(const SBaseArrayDouble2dPtr features, const SArrayDoublePtr labels);

  inline ulong get_n_features() const {
    if (has_data) {
      return features->n_cols();
    } else {
      TICK_ERROR("OnlineForest::get_n_features: the forest has no data yet.")
    }
  }

  inline ulong get_n_samples() const {
    if (has_data) {
      return features->n_rows();
    } else {
      TICK_ERROR("OnlineForest::get_n_samples: the forest has no data yet.")
    }
  }
};

#endif //TICK_ONLINEFOREST_H
