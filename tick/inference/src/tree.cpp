// License: BSD 3 clause

#include "tree.h"

Tree::Tree(OnlineForest &forest) : forest(forest) {
  nodes.emplace_back(std::unique_ptr<Node>(new Node(*this)));
}

void Tree::fit(ulong sample_index) {
  // TODO: Test that the size does not change within successive calls to fit
  std::cout << "Fitting a tree" << std::endl;
}
