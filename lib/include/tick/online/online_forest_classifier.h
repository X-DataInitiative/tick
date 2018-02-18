// License: BSD 3 clause

#ifndef TICK_ONLINE_FOREST_CLASSIFIER_H
#define TICK_ONLINE_FOREST_CLASSIFIER_H

#include <cmath>
#include <iomanip>
#include "tick/base/base.h"
#include "tick/random/rand.h"


// TODO: change the Dirichlet parameter
// TODO: set_feature_importances with a nullptr by default
// TODO: subsample parameter, default 0.5

// TODO: tree aggregation
// TODO: memory optimization (a FeatureSplitter), maximum (sizeof(uint8_t) splits)), a set of current splits
// TODO: only binary features version ?


enum class CriterionClassifier {
  log = 0,
};

enum class FeatureImportanceType {
  no = 0,
  estimated = 1,
  given = 2
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
  float _threshold;
  // Time of creation of the node
  float _time;
  // Range of the features
  ArrayFloat _features_min, _features_max;
  // Number of samples in the node
  uint32_t _n_samples;
  // The label of the sample saved in the node
  float _y_t;
  // Logarithm of the aggregation weight for the node
  float _weight;
  // Logarithm of the aggregation weight for the sub-tree starting at this node
  float _weight_tree;
  // true if the node is a leaf
  bool _is_leaf;
  // Counts the number of sample seen in each class
  ArrayUInt _counts;

 public:
  NodeClassifier(TreeClassifier &tree, uint32_t parent, float time = 0);
  NodeClassifier(const NodeClassifier &node);
  NodeClassifier(const NodeClassifier &&node);
  NodeClassifier &operator=(const NodeClassifier &);
  NodeClassifier &operator=(const NodeClassifier &&) = delete;

  // Computation of log( (e^a + e^b) / 2) in an overproof way
  inline static float log_sum_2_exp(const float a, const float b) {
    // TODO: if |a - b| > 50 skip
    if (a > b) {
      return a + std::log((1 + std::exp(b - a)) / 2);
    } else {
      return b + std::log((1 + std::exp(a - b)) / 2);
    }
  }

  // Update to apply to a node when going forward in the tree (towards leaves)
  float update_downwards(const ArrayDouble &x_t, const double y_t);
  // Update to apply to a node when going upward in the tree (towards the root)
  void update_upwards();
  // Update the prediction of the label
  void update_predict(const double y_t);
  // Update range of the seen features
  void update_range(const ArrayDouble &x_t);
  // Predict function (average of the labels of samples that passed through the node)
  void predict(ArrayDouble &scores) const;
  // Loss function used for aggregation
  float loss(const double y_t);
  // Score of the node when the true label is y
  float score(uint8_t y) const;

  // Get node at index in the tree
  inline NodeClassifier &node(uint32_t index) const;

  // Get number of features
  inline uint32_t n_features() const;
  // Number of classes
  inline uint8_t n_classes() const;
  // Step to use for aggregation
  inline float step() const;
  // Dirichlet prior
  inline float dirichlet() const;
  // Print the node
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
  inline float threshold() const;
  inline NodeClassifier &set_threshold(float threshold);
  inline float time() const;
  inline NodeClassifier &set_time(float time);
  inline float features_min(const uint32_t j) const;
  inline float features_max(const uint32_t j) const;
  inline uint32_t n_samples() const;
  inline bool use_aggregation() const;
  inline float weight() const;
  inline float weight_tree() const;
};

class OnlineForestClassifier;

/*********************************************************************************
 * TreeClassifier
 *********************************************************************************/

class TreeClassifier {
 protected:
  // The forest of the tree
  OnlineForestClassifier &forest;
  // Number of features
  uint32_t _n_features;
  // Number of classes
  uint8_t _n_classes;
  // Number of nodes in the tree
  uint32_t _n_nodes = 0;
  // Iteration counter
  uint32_t iteration = 0;
  // Nodes of the tree
  std::vector<NodeClassifier> nodes = std::vector<NodeClassifier>();

  // Create the root node
  void create_root();
  // Add nodes in the tree
  uint32_t add_node(uint32_t parent, float time = 0);

  // A vector containing estimated feature importances
  ArrayFloat feature_importances_;
  // A vector used for computations
  ArrayFloat intensities;

  void extend_range(uint32_t node_index, const ArrayDouble &x_t, const double y_t);

  uint32_t go_downwards(const ArrayDouble &x_t, double y_t, bool predict);
  void go_upwards(uint32_t leaf_index);

 public:
  TreeClassifier(OnlineForestClassifier &forest);
  TreeClassifier(const TreeClassifier &tree);
  TreeClassifier(const TreeClassifier &&tree);
  TreeClassifier &operator=(const TreeClassifier &) = delete;
  TreeClassifier &operator=(const TreeClassifier &&) = delete;

  // Reserve nodes in the tree in advance
  void reserve_nodes(uint32_t n_nodes);

  void fit(const ArrayDouble &x_t, double y_t);
  void predict(const ArrayDouble &x_t, ArrayDouble &scores, bool use_aggregation);

  // Give the depth of the path for x_t
  uint32_t get_path_depth(const ArrayDouble &x_t);
  // Get the path of x_t
  void get_path(const ArrayDouble &x_t, SArrayUIntPtr path);

  void get_aggregate_path(const SArrayDoublePtr features,
                                SArrayDouble2dPtr node_scores,
                                SArrayDoublePtr aggregation_weights);


    inline uint32_t n_features() const;
  inline uint8_t n_classes() const;
  inline uint32_t n_nodes() const;
  uint32_t n_leaves() const;
  inline float step() const;
  inline float dirichlet() const;

  void print();

  inline CriterionClassifier criterion() const;
  inline bool use_aggregation() const;
  FeatureImportanceType feature_importance_type() const;
  float feature_importance(const uint32_t j) const;
  float given_feature_importance(const uint32_t j) const;

  NodeClassifier &node(uint32_t index) {
    return nodes[index];
  }

  inline ArrayFloat &feature_importances() {
    return feature_importances_;
  }

  static void show_vector(const ArrayDouble x, int precision=2) {
    std::cout << "[";
    for(ulong j = 0; j < x.size(); ++j) {
      std::cout << " " << std::setprecision(precision) << x[j];
    }
    std::cout << " ]" << std::endl;
  }
};

/*********************************************************************************
 * OnlineForestClassifier
 *********************************************************************************/

class OnlineForestClassifier {
 private:
  // Number of features
  uint32_t _n_features;
  // Number of classes in the classification problem
  uint8_t _n_classes;
  // Number of Trees in the forest
  uint8_t _n_trees;
  // Number of passes over each given data
  uint8_t _n_passes;
  // Step-size used for aggregation
  float _step;

  bool _estimate_feature_importances;

  // A vector of given feature importances (not estimated)
  ArrayFloat _given_feature_importances;

  // CriterionClassifier used for splitting (not used for now)
  CriterionClassifier _criterion;
  //
  FeatureImportanceType _feature_importance_type;
  //
  bool _use_aggregation;
  //
  double _subsampling;
  //
  float _dirichlet;
  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;
  // Seed for random number generation
  int _seed;
  // Verbose things or not
  bool _verbose;
  // Iteration counter
  uint32_t _iteration;
  // The list of trees in the forest
  std::vector<TreeClassifier> trees;
  // Random number generator for feature and threshold sampling
  Rand rand;

  // Create trees
  void create_trees();

  // SArrayDouble2dPtr _features;
  // SArrayDoublePtr _labels;

  void check_n_features(uint32_t n_features, bool predict) const;
  inline void check_label(double label) const;

 public:
  OnlineForestClassifier(uint32_t n_features,
                         uint8_t n_classes,
                         uint8_t n_trees,
                         float step,
                         CriterionClassifier criterion,
                         FeatureImportanceType feature_importance_type,
                         bool use_aggregation,
                         float dirichlet,
                         int32_t n_threads,
                         int seed,
                         bool verbose);
  virtual ~OnlineForestClassifier();

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDouble2dPtr scores);

  inline uint32_t sample_feature(const ArrayFloat &prob);
  inline float sample_exponential(float intensity);
  inline float sample_threshold(float left, float right);

  void clear();
  void print();

  uint32_t n_samples() const;
  uint32_t n_features() const;
  uint8_t n_classes() const;

  uint8_t n_trees() const;
  bool use_aggregation() const;
  float step() const;
  OnlineForestClassifier &set_step(const float step);
  float dirichlet() const;
  OnlineForestClassifier &set_dirichlet(const float dirichlet);
  bool verbose() const;
  OnlineForestClassifier &set_verbose(bool verbose);
  CriterionClassifier criterion() const;
  OnlineForestClassifier &set_criterion(CriterionClassifier criterion);
  FeatureImportanceType feature_importance_type() const;

  double given_feature_importances(const ulong j) const;

  int32_t n_threads() const;
  OnlineForestClassifier &set_n_threads(int32_t n_threads);
  int seed() const;
  OnlineForestClassifier &set_seed(int seed);

  void n_nodes(SArrayUIntPtr n_nodes_per_tree);
  void n_leaves(SArrayUIntPtr n_leaves_per_tree);

  OnlineForestClassifier &set_given_feature_importances(const ArrayDouble &feature_importances);

  void get_feature_importances(SArrayDoublePtr feature_importances);

  // Get the path for of a tree for a single vector of features
  // void get_path(uint8_t tree, const SArrayDoublePtr features);

  // Give the depth of the path for x_t
  uint32_t get_path_depth(const uint8_t tree, const SArrayDoublePtr x_t);

  // Get the path of x_t
  void get_path(const uint8_t tree, const SArrayDoublePtr x_t, SArrayUIntPtr path);

};

#endif //TICK_ONLINE_FOREST_CLASSIFIER_H
