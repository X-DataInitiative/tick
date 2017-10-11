#ifndef TICK_NODE_H
#define TICK_NODE_H

// License: BSD 3 clause

#include "base.h"
#include "tree.h"
#include "online_forest.h"

class Node {
 private:
  // Index of the left child
  uint32_t left_child;
  // Index of the right child
  uint32_t right_child;
  // Index of the parent
  uint32_t parent;
  // Creation time of the node (iteration index)
  ulong creation_time;
  // Index of the feature used for the split
  ulong feature;
  // Threshold used for the split
  double threshold;
  // Impurity of the node
  double impurity;
  // Number of samples in the node
  ulong n_samples;
  // Average of the labels in the node (regression only for now)
  double labels_average;
  // Aggregation weight
  double aggregation_weight;
  // Aggregation weight for context-tree weighting
  double aggregation_weight_ctw;
  // Minimum value of each feature (minimum range)
  ArrayDouble features_min;
  // Maximum value of each feature (maximum range)
  ArrayDouble features_max;
  // The indexes (row numbers) of the samples currently in the node
  std::vector samples;
  // The tree of the node
  const Tree &tree;
  // true if the node is a leaf
  bool is_leaf;

 public:
  Node(const Tree &tree);

  inline uint32_t get_left_child() const;
  inline uint32_t get_right_child() const;
  inline uint32_t get_parent() const;
  inline ulong get_creation_time() const;
  inline ulong get_feature() const;
  inline double get_threshold() const;
  inline double get_impurity() const;
  inline ulong get_n_samples() const;
  inline double get_labels_average() const;
  inline double get_aggregation_weight() const;
  inline double get_aggregation_weight_ctw() const;
  inline const ArrayDouble &get_features_min() const;
  inline const ArrayDouble &get_features_max() const;

  inline const Tree &get_tree() const {
    return tree;
  }
};

#endif //TICK_NODE_H
