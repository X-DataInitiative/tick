
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
  // The tree of the node
  Tree &in_tree;

  // Index in the list of nodes of the tree (to be removed later)
  ulong _index;
  // Index of the left child
  ulong _left;
  // Index of the right child
  ulong _right;
  // Index of the parent
  ulong _parent;
  // Creation time of the node (iteration index)
  ulong _creation_time;
  // Index of the feature used for the split
  ulong _feature = 0;
  // Threshold used for the split
  double _threshold = 0;
  // Impurity of the node
  double _impurity = 0;
  // Number of samples in the node
  ulong _n_samples;
  // Average of the labels in the node (regression only for now)
  double _labels_average = 0;
  // Aggregation weight
  double _aggregation_weight = 1;
  // Aggregation weight for context-tree weighting
  double _aggregation_weight_ctw = 1;
  // Minimum value of each feature (minimum range)
  ArrayDouble _features_min;
  // Maximum value of each feature (maximum range)
  ArrayDouble _features_max;
  // The indexes (row numbers) of the samples currently in the node
  // std::vector<ulong> samples;
  // true if the node is a leaf
  bool _is_leaf = true;

 public:
  Node(Tree &tree, ulong index, ulong parent, ulong creation_time);

  Node(const Node &node);

  Node(const Node &&node);

  ~Node() {
    // std::cout << "~Node()\n";
  }

  Node &operator=(const Node &) = delete;
  Node &operator=(const Node &&) = delete;

  inline ulong index() const {
    return _index;
  }

  inline ulong left() const {
    return _left;
  }

  inline Node &set_left(ulong left) {
    _left = left;
    return *this;
  }

  inline ulong right() const {
    return _right;
  }

  inline Node &set_right(ulong right) {
    _right = right;
    return *this;
  }

  inline const bool is_leaf() const {
    return _is_leaf;
  }

  inline Node &set_is_leaf(bool is_leaf) {
    _is_leaf = is_leaf;
    return *this;
  }

  inline ulong parent() const {
    return _parent;
  }

  inline ulong creation_time() const {
    return _creation_time;
  }

  inline ulong feature() const {
    return _feature;
  }

  inline double threshold() const {
    return _threshold;
  }

  inline double impurity() const {
    return _impurity;
  }

  inline ulong n_samples() const {
    return _n_samples;
  }

  inline double labels_average() const {
    return _labels_average;
  }

  inline double aggregation_weight() const {
    return _aggregation_weight;
  }

  inline double aggregation_weight_ctw() const {
    return _aggregation_weight_ctw;
  }

  inline const ArrayDouble &features_min() const {
    return _features_min;
  }

  inline const ArrayDouble &features_max() const {
    return _features_max;
  }

  inline Node &set_features_min(const ArrayDouble &features_min) {
    _features_min = features_min;
    return *this;
  }

  inline Node &set_features_max(const ArrayDouble &features_max) {
    _features_max = features_max;
    return *this;
  }

  inline Tree &get_tree() const {
    return in_tree;
  }

  // Update the statistics of the node using the sample
  void update(ulong sample_index);

  inline ArrayDouble get_features(ulong sample_index) const;

  ulong n_features() const;

  inline double get_label(ulong sample_index) const;

  void print() {

    std::cout << "Node(i: " << _index << ", p: " << _parent
              << ", l: " << _left
              << ", r: " << _right
              << ", n: " << n_samples()
              << ", i: " << _is_leaf
              << ", t: " << _creation_time
              << ", avg: " << std::setprecision(2) << _labels_average
              << ", feat_min=[" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
              << _features_min[1] << "]"
              << ", feat_max=[" << std::setprecision(2) <<_features_max[0] << ", " << std::setprecision(2)
              << _features_max[1] << "]"
              << ")\n";
  }
};

class OnlineForest;

class Tree {
  friend class Node;

 private:
  bool already_fitted = false;

  // The list of nodes in the tree
  std::vector<Node> nodes = std::vector<Node>();

  // Number of nodes in the tree
  ulong n_nodes = 0;

  // Depth of the tree
  // ulong depth = 0;

  // The forest of the tree
  OnlineForest &forest;

  // Iteration counter
  ulong iteration = 0;

  // Split the node at given index
  void split_node(ulong index);

  ulong add_node(ulong parent, ulong creation_time);

 public:
  Tree(OnlineForest &forest);

  Tree(const Tree &tree);
  Tree(const Tree &&tree);

  Tree &operator=(const Tree &) = delete;
  Tree &operator=(const Tree &&) = delete;

  // Launch a pass on the given data
  void fit(ulong sample_index);

  inline OnlineForest &get_forest() const;

  inline Node &get_root() {
    return nodes[0];
  }

  inline Node &get_node(ulong node_index);

  inline ulong n_features() const;

  inline ArrayDouble get_features(ulong sample_index) const;

  inline double get_label(ulong sample_index) const;

  ~Tree() {
    std::cout << "~Tree()\n";
  }

  void print() {
    std::cout << "Tree(n_nodes: " << n_nodes << ", iteration: " << iteration << std::endl;
    for (Node &node: nodes) {
      std::cout << "    ";
      node.print();
    }
    std::cout << ")" << std::endl;
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
  uint32_t _n_min_samples;
  // Number of candidate splits to be considered
  uint32_t _n_splits;
  // Iteration counter
  ulong t;
  // The vector of features
  SArrayDouble2dPtr _features;
  // The vector of labels
  SArrayDoublePtr _labels;

  // The list of trees in the forest
  std::vector<Tree> trees;

  // ulong n_features;

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

  // Returns a uniform integer in the set {0, ..., m - 1}
  inline ulong rand_unif(ulong m) {
    return rand.uniform_int(ulong{0}, m);
  }

  // Fit the forest by doing a certain number number of iterations
  void fit(ulong n_iter = 0);

  // Pass the data to the forest
  void set_data(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

  inline ulong n_features() const {
    if (has_data) {
      return _features->n_cols();
    } else {
      TICK_ERROR("OnlineForest::get_n_features: the forest has no data yet.")
    }
  }

  inline ulong n_samples() const {
    if (has_data) {
      return _features->n_rows();
    } else {
      TICK_ERROR("OnlineForest::get_n_samples: the forest has no data yet.")
    }
  }

  inline uint32_t n_min_samples() const {
    return _n_min_samples;
  }

  inline uint32_t n_splits() const {
    return _n_splits;
  }

  inline ArrayDouble features(ulong sample_index) const {
    return view_row(*_features, sample_index);
  }

  inline double label(ulong i) const {
    return (*_labels)[i];
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
