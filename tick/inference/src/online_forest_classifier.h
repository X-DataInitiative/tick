
#ifndef TICK_ONLINE_FOREST_CLASSIFIER_H
#define TICK_ONLINE_FOREST_CLASSIFIER_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"


// TODO: in the forest, tests for the input labels and the size of the features (for fit and predict_proba and predict)


enum class CriterionClassifier {
  log = 0,
};

class TreeClassifier;

/*********************************************************************************
 * NodeClassifier
 *********************************************************************************/

class NodeClassifier {
 protected:
  // Tree containing the node
  TreeClassifier &_tree;
  // Index of the parent
  uint32_t _parent;
  // Index of the left child
  uint32_t _left;
  // Index of the right child
  uint32_t _right;
  // Index of the feature used for the split
  uint32_t _feature;
  // Threshold used for the split
  double _threshold;
  // Time of creation of the node
  double _time;
  // Range of the features
  ArrayDouble _features_min;
  ArrayDouble _features_max;
  // Number of samples in the node
  uint32_t _n_samples;
  // The features of the sample saved in the node
  // TODO: use a unique_ptr on x_t
  ArrayDouble _x_t;
  // The label of the sample saved in the node
  double _y_t;
  // Logarithm of the aggregation weight for the node
  double _weight;
  // Logarithm of the agregation weight for the sub-tree starting at this node
  double _weight_tree;
  // true if the node is a leaf
  bool _is_leaf;
  // Counts the number of sample seen in each class
  ArrayULong _counts;

 public:
  NodeClassifier(TreeClassifier &tree, uint32_t parent, double time = 0);
  NodeClassifier(const NodeClassifier &node);
  NodeClassifier(const NodeClassifier &&node);
  NodeClassifier &operator=(const NodeClassifier &);
  NodeClassifier &operator=(const NodeClassifier &&) = delete;

  // Computation of log( (e^a + e^b) / 2) in an overproof way
  inline static double log_sum_2_exp(const double a, const double b) {
    // TODO if |a - b| > 50 skip
    if (a > b) {
      return a + std::log((1 + std::exp(b - a)) / 2);
    } else {
      return b + std::log((1 + std::exp(a - b)) / 2);
    }
  }

  // Update to apply to a node when going forward in the tree (towards leaves)
  void update_downwards(const ArrayDouble &x_t, const double y_t);
  // Update to apply to a node when going upward in the tree (towards the root)
  void update_upwards();
  // Update the prediction of the label
  void update_predict(const double y_t);
  // Update range of the seen features
  void update_range(const ArrayDouble &x_t);
  // Predict function (average of the labels of samples that passed through the node)
  void predict(ArrayDouble &scores) const;
  // Loss function used for aggregation

  double score(uint8_t y) const;

  double loss(const double y_t);

  bool is_same(const ArrayDouble &x_t);

  // Get node at index in the tree
  inline NodeClassifier &node(uint32_t index) const;
  // Get number of features
  inline uint32_t n_features() const;
  // Number of classes
  inline uint8_t n_classes() const;
  // Step to use for aggrgation
  inline double step() const;
  // Print of the node
  void print();

  inline uint32_t parent() const;
  inline NodeClassifier &set_parent(uint32_t parent);
  inline uint32_t left() const;
  inline NodeClassifier &set_left(uint32_t left);
  inline uint32_t right() const;
  inline NodeClassifier &set_right(uint32_t right);
  inline bool is_leaf() const;
  inline NodeClassifier &set_is_leaf(bool is_leaf);
  inline uint32_t feature() const;
  inline NodeClassifier &set_feature(uint32_t feature);
  inline double threshold() const;
  inline NodeClassifier &set_threshold(double threshold);
  inline double time() const;
  inline NodeClassifier &set_time(double time);
  inline double features_min(const uint32_t j) const;
  inline NodeClassifier &set_features_min(const ArrayDouble &features_min);
  inline double features_max(const uint32_t j) const;
  inline NodeClassifier &set_features_max(const ArrayDouble &features_max);
  inline uint32_t n_samples() const;
  inline NodeClassifier &set_n_samples(uint32_t n_samples);
  inline double weight() const;
  inline NodeClassifier &set_weight(double weight);
  inline double weight_tree() const;
  inline NodeClassifier &set_weight_tree(double weight);
  inline const ArrayDouble &x_t() const;
  inline NodeClassifier &set_x_t(const ArrayDouble &x_t);
  inline double y_t() const;
  inline NodeClassifier &set_y_t(const double y_t);
};

class OnlineForestClassifier;

/*********************************************************************************
 * TreeClassifier
 *********************************************************************************/

class TreeClassifier {
 protected:
  // The forest of the tree
  OnlineForestClassifier &forest;
  // Number of nodes in the tree
  uint32_t _n_nodes = 0;
  // Iteration counter
  uint32_t iteration = 0;
  // Nodes of the tree
  std::vector<NodeClassifier> nodes = std::vector<NodeClassifier>();
  // Split the node at given index
  // uint32_t split_leaf(uint32_t index, const ArrayDouble &x_t, double y_t);
  // Add nodes in the tree
  uint32_t add_node(uint32_t parent, double time = 0);

  void extend_range(uint32_t node_index, const ArrayDouble &x_t, const double y_t);

  uint32_t go_downwards(const ArrayDouble &x_t, double y_t, bool predict);
  void go_upwards(uint32_t leaf_index);

 public:
  TreeClassifier(OnlineForestClassifier &forest);
  TreeClassifier(const TreeClassifier &tree);
  TreeClassifier(const TreeClassifier &&tree);
  TreeClassifier &operator=(const TreeClassifier &) = delete;
  TreeClassifier &operator=(const TreeClassifier &&) = delete;

  void fit(const ArrayDouble &x_t, double y_t);
  void predict(const ArrayDouble &x_t, ArrayDouble &scores, bool use_aggregation);

  inline uint32_t n_features() const;
  inline uint8_t n_classes() const;
  inline uint32_t n_nodes() const;
  uint32_t n_leaves() const;
  inline double step() const;

  void print() {
    std::cout << "Tree(n_nodes: " << _n_nodes << std::endl;
    std::cout << " ";
    uint32_t index = 0;
    for (NodeClassifier &node : nodes) {
      std::cout << "index: " << index << " ";
      node.print();
      index++;
    }
    std::cout << ")" << std::endl;
  }

  inline CriterionClassifier criterion() const;

  NodeClassifier &node(uint32_t index) {
    return nodes[index];
  }
};

/*********************************************************************************
 * OnlineForestClassifier
 *********************************************************************************/

class OnlineForestClassifier {
 private:
  // Number of classes in the classification problem
  uint8_t _n_classes;
  // Number of Trees in the forest
  uint32_t _n_trees;
  // Step-size used for aggregation
  double _step;
  // CriterionClassifier used for splitting (not used for now)
  CriterionClassifier _criterion;
  //
  bool _use_aggregation;
  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;
  // Seed for random number generation
  int _seed;
  // Verbose things or not
  bool _verbose;
  // Number of features.
  uint32_t _n_features;
  // Iteration counter
  uint32_t _iteration;
  // The list of trees in the forest
  std::vector<TreeClassifier> trees;
  // Random number generator for feature and threshold sampling
  Rand rand;

  ArrayDouble _feature_importances;
  // Create trees
  void create_trees();

  SArrayDouble2dPtr _features;
  SArrayDoublePtr _labels;

 public:
  OnlineForestClassifier(uint8_t n_classes, uint32_t n_trees, double step = 1.0,
                         CriterionClassifier criterion = CriterionClassifier::log,
                         bool use_aggregation = true, int32_t n_threads = 1,
                         int seed = 0, bool verbose = false);
  virtual ~OnlineForestClassifier();

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDouble2dPtr scores);

  inline uint32_t sample_feature();
  inline uint32_t sample_feature(const ArrayDouble &prob);

  inline uint32_t sample_feature_bis();

  inline double sample_exponential(double intensity);

  inline double sample_threshold(double left, double right);

  void clear();

  inline double step() const {
    return _step;
  }

  void print() {
    for (TreeClassifier &tree: trees) {
      tree.print();
    }
  }

  inline uint32_t n_samples() const {
    if (_iteration > 0) {
      return _iteration;
    } else {
      TICK_ERROR("You must call ``fit`` before asking for ``n_samples``.")
    }
  }

  inline uint32_t n_features() const {
    if (_iteration > 0) {
      return _n_features;
    } else {
      TICK_ERROR("You must call ``fit`` before asking for ``n_features``.")
    }
  }

  inline uint8_t n_classes() const {
    return _n_classes;
  }

  OnlineForestClassifier &set_n_classes(uint8_t n_classes) {
    if (_iteration == 0) {
      _n_classes = n_classes;
    } else {
      TICK_ERROR("OnlineForest::set_n_classes can be called only once !")
    }
    return *this;
  }

  inline OnlineForestClassifier &set_n_features(uint32_t n_features) {
    if (_iteration == 0) {
      _n_features = n_features;
    }
    return *this;
  }

  inline uint32_t n_trees() const {
    return _n_trees;
  }

  inline OnlineForestClassifier &set_n_trees(uint32_t n_trees) {
    _n_trees = n_trees;
    return *this;
  }

  inline int32_t n_threads() const {
    return _n_threads;
  }

  inline CriterionClassifier criterion() const {
    return _criterion;
  }

  inline int seed() const {
    return _seed;
  }

  inline OnlineForestClassifier &set_seed(int seed) {
    _seed = seed;
    rand.reseed(seed);
    return *this;
  }

  OnlineForestClassifier &set_n_threads(int32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

  inline OnlineForestClassifier &set_criterion(CriterionClassifier criterion) {
    _criterion = criterion;
    return *this;
  }

  inline void set_feature_importances(const ArrayDouble &feature_importances) {
    _feature_importances = feature_importances;
  }

//  inline bool verbose() const;
//  inline OnlineForestClassifier &set_verbose(bool verbose);
};

#endif //TICK_ONLINE_FOREST_CLASSIFIER_H
