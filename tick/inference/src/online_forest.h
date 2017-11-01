
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"


enum class Criterion {
  unif = 0,
  mse
};


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
  std::vector<ulong> _samples;
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

  inline Node& set_feature(ulong feature) {
    _feature = feature;
    return *this;
  }

  inline double threshold() const {
    return _threshold;
  }

  inline Node& set_threshold(double threshold) {
    _threshold = threshold;
    return *this;
  }

  inline double impurity() const {
    return _impurity;
  }

  inline ulong n_samples() const {
    return _n_samples;
  }

  inline Node& set_n_samples(ulong n_samples) {
    _n_samples = n_samples;
    return *this;
  }

  inline double labels_average() const {
    return _labels_average;
  }

  inline Node& set_labels_average(double avg) {
    _labels_average = avg;
    return *this;
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

  inline Node &set_features_min(const ArrayDouble &features_min) {
    _features_min = features_min;
    return *this;
  }

  inline Node &set_features_min(const ulong j, const double x) {
    _features_min[j] = x;
    return *this;
  }

  inline const ArrayDouble &features_max() const {
    return _features_max;
  }

  inline Node &set_features_max(const ulong j, const double x) {
    _features_max[j] = x;
    return *this;
  }

  inline Node &set_features_max(const ArrayDouble &features_max) {
    _features_max = features_max;
    return *this;
  }

  inline const std::vector<ulong>& samples() const {
    return _samples;
  }

  inline ulong sample(ulong index) const {
    return _samples[index];
  }

  inline Node& add_sample(ulong index) {
    _samples.push_back(index);
    return *this;
  }

  inline Tree &get_tree() const {
    return in_tree;
  }

  // Update the statistics of the node using the sample
  void update(ulong sample_index, bool update_range=true);

  inline ArrayDouble get_features(ulong sample_index) const;

  ulong n_features() const;

  inline double get_label(ulong sample_index) const;

  void print() {

    std::cout << "Node(i: " << _index << ", p: " << _parent
              << ", f: " << _feature
              << ", th: " << _threshold
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

  ulong find_leaf(ulong sample_index, bool predict);

 public:
  Tree(OnlineForest &forest);

  Tree(const Tree &tree);
  Tree(const Tree &&tree);

  Tree &operator=(const Tree &) = delete;
  Tree &operator=(const Tree &&) = delete;

  // Launch a pass on the given data
  void fit(ulong sample_index);

  double predict(ulong sample_index);

  inline OnlineForest &get_forest() const;

  inline Node &get_root() {
    return nodes[0];
  }

  inline const Node &get_node(ulong node_index) const;

  inline ulong n_features() const;

  inline uint32_t min_samples_split() const;

  inline ulong sample_feature_uniform();

  inline double sample_threshold_uniform(double left, double right);

  inline ArrayDouble get_features(ulong sample_index) const;

  inline ArrayDouble get_features_predict(ulong sample_index) const;

  inline double get_label(ulong sample_index) const;

  ~Tree() {
    // std::cout << "~Tree()\n";
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
  uint32_t _n_trees;

  // Number of threads to use for parallel growing of trees
  uint32_t _n_threads;

  // Number of samples required in a node before splitting it (this means that we wait to have
  // n_min_samples in the node before computing the candidate splits and impurities, which uses
  // the range of the samples)
  uint32_t _min_samples_split;

  // Number of candidate splits to be considered
  uint32_t _n_splits;

  Criterion _criterion;

  int32_t _max_depth;

  int _seed;

  bool _verbose;

  bool _warm_start;

  // Iteration counter
  ulong t;
  // The matrix of features used for fitting
  SArrayDouble2dPtr _features_fit;
  // The vector of labels used for fitting
  SArrayDoublePtr _labels_fit;

  // The vector of features used for prediction
  SArrayDouble2dPtr _features_predict;

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

  // Random number generator for feature sampling
  Rand rand_feature;

  // Random number generator for feature sampling
  Rand rand_threshold;

  ulong get_next_sample();

  void shuffle();

 public:
  OnlineForest(uint32_t n_trees,
               Criterion criterion,
               int32_t max_depth,
               uint32_t min_samples_split,
               uint32_t n_threads,
               int seed,
               bool verbose,
               bool warm_start,
               uint32_t n_splits);

  ~OnlineForest() {
    // std::cout << "~OnlineForest()\n";
  }

  // Returns a uniform integer in the set {0, ..., m - 1}
  inline ulong rand_unif(ulong m) {
    return rand.uniform_int(ulong{0}, m);
  }

  inline ulong sample_feature_uniform() {
    return rand_feature.uniform_int(ulong{0}, n_features() - 1);
  }

  inline double sample_threshold_uniform(double left, double right) {
    return rand_threshold.uniform(left, right);
  }

  // Fit the forest by doing a certain number number of iterations
  void fit(ulong n_iter = 0);

  // Pass the data to the forest
  void set_data(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions);

    inline ulong n_features() const {
    if (has_data) {
      return _features_fit->n_cols();
    } else {
      TICK_ERROR("OnlineForest::get_n_features: the forest has no data yet.")
    }
  }

  inline ulong n_samples() const {
    if (has_data) {
      return _features_fit->n_rows();
    } else {
      TICK_ERROR("OnlineForest::get_n_samples: the forest has no data yet.")
    }
  }

  inline ArrayDouble features(ulong sample_index) const {
    return view_row(*_features_fit, sample_index);
  }

  inline ArrayDouble features_predict(ulong sample_index) const {
    return view_row(*_features_predict, sample_index);
  }

  inline double label(ulong i) const {
    return (*_labels_fit)[i];
  }

  void print() {
    std::cout << "Forest" << std::endl;
    for (Tree &tree: trees) {
      tree.print();
    }
  }

  inline uint32_t n_trees() const {
    return _n_trees;
  }

  inline OnlineForest& set_n_trees(uint32_t n_trees) {
    _n_trees = n_trees;
    return *this;
  }

  inline uint32_t n_splits() const {
    return _n_splits;
  }

  inline OnlineForest& set_n_splits(uint32_t n_splits) {
    _n_splits = n_splits;
    return *this;
  }

  inline uint32_t n_threads() const {
    return _n_threads;
  }

  inline OnlineForest& set_n_threads(uint32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

  inline uint32_t min_samples_split() const {
    return _min_samples_split;
  }

  inline OnlineForest& set_min_samples_split(uint32_t min_samples_split) {
    _min_samples_split = min_samples_split;
    return *this;
  }

  inline Criterion criterion() const {
    return _criterion;
  }

  inline OnlineForest& set_criterion(Criterion criterion) {
    _criterion = criterion;
    return *this;
  }

  inline int32_t max_depth() const {
    return _max_depth;
  }

  inline OnlineForest& set_max_depth(int32_t max_depth) {
    _max_depth = max_depth;
    return *this;
  }

  inline int seed() const {
    return _seed;
  }

  inline OnlineForest& set_seed(int seed) {
    _seed = seed;
    rand.reseed(seed);
    rand_feature.reseed(seed);
    rand_threshold.reseed(seed);
    return *this;
  }

  inline bool verbose() const {
    return _verbose;
  }

  inline OnlineForest& set_verbose(bool verbose) {
    _verbose = verbose;
    return *this;
  }

  inline bool warm_start() const {
    return _warm_start;
  }

  inline OnlineForest& set_warm_start(bool warm_start) {
    _warm_start = warm_start;
    return *this;
  }

  ulong get_t() {
    return t;
  }
};

#endif //TICK_ONLINEFOREST_H
