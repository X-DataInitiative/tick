// License: BSD 3 clause

#include "node.h"

Node::Node(Tree &tree) : tree(tree) {
  // At its creation, a node is a leaf
  is_leaf = true;
}

uint32_t Node::get_left_child() const {
  return left_child;
}
uint32_t Node::get_right_child() const {
  return right_child;
}
uint32_t Node::get_parent() const {
  return parent;
}
ulong Node::get_creation_time() const {
  return creation_time;
}
ulong Node::get_feature() const {
  return feature;
}
double Node::get_threshold() const {
  return threshold;
}
double Node::get_impurity() const {
  return impurity;
}
ulong Node::get_n_samples() const {
  return n_samples;
}
double Node::get_labels_average() const {
  return labels_average;
}
double Node::get_aggregation_weight() const {
  return aggregation_weight;
}
double Node::get_aggregation_weight_ctw() const {
  return aggregation_weight_ctw;
}
const ArrayDouble &Node::get_features_min() const {
  return features_min;
}
const ArrayDouble &Node::get_features_max() const {
  return features_max;
}
