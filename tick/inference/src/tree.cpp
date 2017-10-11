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
