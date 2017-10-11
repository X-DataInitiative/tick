#ifndef TICK_TREE_H
#define TICK_TREE_H

// License: BSD 3 clause

#include "base.h"
#include "node.h"
#include "online_forest.h"

class OnlineForest;

class Tree {
 private:
  bool already_fitted;

  // Number of samples required in a node before splitting it (this means that we wait to have
  // n_min_samples in the node before computing the candidate splits and impurities, which uses
  // the range of the samples)
  uint32_t n_min_samples;

  // Number of candidate splits to be considered
  uint8_t n_splits;

  // The list of nodes in the tree
  std::vector<Node> nodes;

  // The forest of the tree
  const OnlineForest &forest;

 public:
  Tree(const OnlineForest &forest);

  // Launch a pass on the given data
  void fit(ulong n_iter = 0);

  inline const OnlineForest &get_forest() const {
    return forest;
  }
};

#endif //TICK_TREE_H
