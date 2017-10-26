/*
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"

// Forward declaration of a Tree
class Tree;

class Node {
 private:
  // Index of the left child
  ulong left_child;
  // Index of the right child
  ulong right_child;
  // Index of the parent
  ulong parent;
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
  std::vector<ulong> samples;
  // The tree of the node
  Tree &tree;
  // true if the node is a leaf
  bool is_leaf;

 public:
  Node(Tree &tree);

  Node(Tree &tree, uint32_t parent, ulong creation_time);

  Node(const Node &node);

  ~Node() {
    std::cout << "~Node()\n";
  }

  inline uint32_t get_left_child() const {
    return left_child;
  }

  inline uint32_t get_right_child() const {
    return right_child;
  }

  inline uint32_t get_parent() const {
    return parent;
  }

  inline ulong get_creation_time() const {
    return creation_time;
  }

  inline ulong get_feature() const {
    return feature;
  }

  inline double get_threshold() const {
    return threshold;
  }

  inline double get_impurity() const {
    return impurity;
  }

  inline ulong get_n_samples() const {
    return n_samples;
  }

  inline double get_labels_average() const {
    return labels_average;
  }

  inline double get_aggregation_weight() const {
    return aggregation_weight;
  }

  inline double get_aggregation_weight_ctw() const {
    return aggregation_weight_ctw;
  }

  inline const ArrayDouble &get_features_min() const {
    return features_min;
  }

  inline const ArrayDouble &get_features_max() const {
    return features_max;
  }

  inline void set_features_min(const ArrayDouble &features_min) {
    std::cout << "in set_features_min(const ArrayDouble &features_min)" << std::endl;
    this->features_min.print();
    features_min.print();
    this->features_min = features_min;
  }

  inline void set_features_max(const ArrayDouble &features_max) {
    this->features_max = features_max;
  }

  inline const bool get_is_leaf() const {
    return is_leaf;
  }

  inline Tree &get_tree() const {
    return tree;
  }

  // Update the statistics of the node using the sample
  void update(ulong sample_index, bool update_range = false);

  // Split the node
  void split(ulong node_index, uint32_t n_splits);

  inline ArrayDouble get_features(ulong sample_index) const;

  ulong get_n_features() const;

  inline double get_label(ulong sample_index) const;

  void print() {

    std::cout << "Node(parent: " << parent
              << ", left_child: " << left_child
              << ", right_child: " << right_child
              << ", n_samples=" << n_samples
              << ", feat_min=[" << std::setprecision(3) << features_min[0] << ", " << std::setprecision(3)
              << features_min[1] << "]"
              << ", feat_max=[" << std::setprecision(3) << features_max[0] << ", " << std::setprecision(3)
              << features_max[1] << "]"
              << ")\n";
  }
};

class OnlineForest;

class Tree {
 private:
  bool already_fitted;

  // The list of nodes in the tree
  std::vector<Node> nodes;

  // The forest of the tree
  OnlineForest &forest;

 public:
  Tree(OnlineForest &forest);

  Tree(const Tree & tree);

  // Launch a pass on the given data
  void fit(ulong n_iter = 0);

  inline OnlineForest &get_forest() const;

  inline Node &get_root() {
    return nodes[0];
  }

  ulong add_node();

  ulong add_node(uint32_t parent, ulong creation_time);

  inline Node &get_node(ulong node_index);

  inline ulong get_n_features() const;

  inline ArrayDouble get_features(ulong sample_index) const;

  inline double get_label(ulong sample_index) const;

  ~Tree() {
    std::cout << "~Tree()\n";
  }


  void print() {
    std::cout << "Tree" << std::endl;
    for (Node &node: nodes) {
      node.print();
    }
  }
};

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
  uint32_t n_splits;
  // Iteration counter
  ulong t;
  // The vector of features
  SArrayDouble2dPtr features;
  // The vector of labels
  SArrayDoublePtr labels;

  // The list of trees in the forest
  std::vector<Tree> trees;

  ulong n_features;

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

  Rand rand;

  ulong get_next_sample();

  void shuffle();

 public:
  OnlineForest(uint32_t n_trees,
               uint32_t n_min_samples,
               uint32_t n_splits);

  ~OnlineForest() {
    std::cout << "~OnlineForest()\n";
  }

  OnlineForest(const OnlineForest & forest);

  // Returns a uniform integer in the set {0, ..., m - 1}
  inline ulong rand_unif(ulong m) {
    return rand.uniform_int(ulong{0}, m);
  }

  // Fit the forest by doing a certain number number of iterations
  void fit(ulong n_iter = 0);

  // Pass the data to the forest
  void set_data(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

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

  inline uint32_t get_n_min_samples() const {
    return n_min_samples;
  }

  inline uint32_t get_n_splits() const {
    return n_splits;
  }

  inline ArrayDouble get_features(ulong sample_index) const {
    return view_row(*features, sample_index);
  }

  inline double get_label(ulong i) const {
    return (*labels)[i];
  }

  void print() {
    std::cout << "Forest" << std::endl;
    for (Tree &tree: trees) {
      tree.print();
    }
  }

  ulong get_t() {
    return t;
  }
};

#endif //TICK_ONLINEFOREST_H
*/


#include <memory>
#include <vector>

class Node;

class Tree {
  friend class Node;

 private:

  size_t m_count = 0;
  std::vector<Node> nodes;

  Tree(const Tree &o) = delete;
  Tree(const Tree &&o) = delete;
  Tree &operator=(const Tree &) = delete;
  Tree &operator=(const Tree &&) = delete;

 public:
  Tree();

  size_t insert(const uint32_t &_v);

  // Returns the root of the tree
  Node &root() {
    return nodes[0];
  }

  // Returns the node at given index in the tree
  Node &node(size_t index) {
    return nodes[index];
  }

  size_t get_next() {
    return m_count++;
  }

  // Number of nodes in the tree
  size_t size() {
    return nodes.size();
  }
};

class Node {
 private:
  Tree &m_tree;
  size_t m_value = 0;
  size_t i_index = 0, i_left = 0, i_right = 0, i_parent;

 public:
  Node(Tree &tree) : m_tree(tree) {}

  Node(Tree &tree, size_t _value, size_t _i)
      : m_tree(tree), m_value(_value), i_index(_i) {
  }

  Node(const Node &o) : m_tree(o.m_tree) {
    m_value = o.m_value;
    i_left = o.i_left;
    i_right = o.i_right;
    i_index = o.i_index;
  }
  Node(const Node &&o) : m_tree(o.m_tree) {
    m_value = o.m_value;
    i_left = o.i_left;
    i_right = o.i_right;
    i_index = o.i_index;
  }
  ~Node() {}
  Node &operator=(const Node &) = delete;
  Node &operator=(const Node &&) = delete;

  void fill() {
    size_t index = i_index;
    Tree *tree = &m_tree;
    tree->insert(1);
    tree->nodes[tree->m_count].i_parent = index;

    tree->insert(1);
    tree->nodes[tree->m_count].i_parent = index;

    tree->nodes[index].m_value = 0;
  }

  bool has_parent() { return i_index; }
  size_t left() { return i_left; }
  size_t right() { return i_right; }
  size_t value() { return m_value; }
  size_t parent() { return i_parent; }
};

Tree::Tree() {
  nodes.emplace_back(*this, 1, m_count++);
}

size_t Tree::insert(const uint32_t &_v) {
  nodes.emplace_back(*this, _v, m_count);
  return m_count++;
}

int main(int argc, char *argv[]) {

  // auto now = kul::Now::NANOS();

  Tree tree;
  Node &root(tree.root());
  root.fill();

  size_t i_left = root.left(), i_right = root.right();
  for (size_t i = 0; i < 654321; i++) {
    tree.node(i_left).fill();
    i_left = tree.node(i_left).left();
    tree.node(i_right).fill();
    i_right = tree.node(i_right).left();

  }
  while (tree.node(i_left).has_parent()) i_left = tree.node(i_left).parent();
  while (tree.node(i_right).has_parent()) i_right = tree.node(i_right).parent();

//   KLOG(INF) << "time in millis : " << (kul::Now::NANOS() - now) / 1e6;
  //KLOG(INF) << tree.size();

  return 0;
}