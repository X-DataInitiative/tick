// License: BSD 3 clause

#include "tree.h"

Tree::Tree(const OnlineForest &forest) : forest(forest) {
  // At creation of the tree, we create the first root node
  nodes.emplace_back(this);
}

void Tree::fit(ulong sample_index) {
  // TODO: Test that the size does not change within successive calls to fit
  std::cout << "Fitting a tree" << std::endl;
}

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
