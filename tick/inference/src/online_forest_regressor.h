
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"


// TODO: faire tres attention au features binaires si le range est 0 sur toutes les coordonnées, ne rien faire
// TODO: code a classifier

// TODO: choisir la feature proportionnellement au ratio des range de features, mais attention au cas de features
//       discretes
// TODO: une option pour créer une cellule vide, enfin oublier les donnes dans la cellule quand elle a ete splitee

// TODO: choix de la feature les labels

// TODO: pour la classification, on utilise pas les frequences, on utilise des frequences regularisees, prior Dirichlet p_c = (n_c + 0.5) + (\sum n_c + C / 2). En fait une option

// TODO: check that not using reserve in the forest works as well...


enum class CriterionRegressor {
  unif = 0,
  mse
};



class TreeRegressor;

/*********************************************************************************
 * NodeRegressor
 *********************************************************************************/

class NodeRegressor {
 protected:
  // Tree containing the node
  TreeRegressor &_tree;
  // Index of the parent
  ulong _parent;
  // Index of the left child
  ulong _left;
  // Index of the right child
  ulong _right;
  // Index of the feature used for the split
  ulong _feature;
  // Threshold used for the split
  double _threshold;
  // Number of samples in the node
  ulong _n_samples;
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
  // Average of the labels in the node (regression only for now)
  double _predict = 0;

 public:
  NodeRegressor(TreeRegressor &tree, ulong parent);
  NodeRegressor(const NodeRegressor &node);
  NodeRegressor(const NodeRegressor &&node);
  NodeRegressor &operator=(const NodeRegressor &) = delete;
  NodeRegressor &operator=(const NodeRegressor &&) = delete;

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
  // Predict function (average of the labels of samples that passed through the node)
  double predict() const;
  // Loss function used for aggregation
  double loss(const double y_t);
  // Get node at index in the tree
  inline NodeRegressor &node(ulong index) const;
  // Get number of features
  inline ulong n_features() const;
  // Step to use for aggrgation
  inline double step() const;
  // Print of the node
  void print();

  inline ulong parent() const;
  inline ulong left() const;
  inline NodeRegressor &set_left(ulong left);
  inline ulong right() const;
  inline NodeRegressor &set_right(ulong right);
  inline bool is_leaf() const;
  inline NodeRegressor &set_is_leaf(bool is_leaf);
  inline ulong feature() const;
  inline NodeRegressor &set_feature(ulong feature);
  inline double threshold() const;
  inline NodeRegressor &set_threshold(double threshold);
  inline ulong n_samples() const;
  inline NodeRegressor &set_n_samples(ulong n_samples);
  inline double weight() const;
  inline NodeRegressor &set_weight(double weight);
  inline double weight_tree() const;
  inline NodeRegressor &set_weight_tree(double weight);
  inline const ArrayDouble &x_t() const;
  inline NodeRegressor &set_x_t(const ArrayDouble &x_t);
  inline double y_t() const;
  inline NodeRegressor &set_y_t(const double y_t);
};

class OnlineForestRegressor;

/*********************************************************************************
 * TreeRegressor
 *********************************************************************************/

class TreeRegressor {
 protected:
  // The forest of the tree
  OnlineForestRegressor &forest;
  // Number of nodes in the tree
  ulong _n_nodes = 0;
  // Iteration counter
  ulong iteration = 0;
  // Nodes of the tree
  std::vector<NodeRegressor> nodes = std::vector<NodeRegressor>();
  // Split the node at given index
  ulong split_leaf(ulong index, const ArrayDouble &x_t, double y_t);
  // Add nodes in the tree
  ulong add_node(ulong parent);

  ulong go_downwards(const ArrayDouble &x_t, double y_t, bool predict);
  void go_upwards(ulong leaf_index);

 public:
  TreeRegressor(OnlineForestRegressor &forest);
  TreeRegressor(const TreeRegressor &tree);
  TreeRegressor(const TreeRegressor &&tree);
  TreeRegressor &operator=(const TreeRegressor &) = delete;
  TreeRegressor &operator=(const TreeRegressor &&) = delete;

  void fit(const ArrayDouble &x_t, double y_t);
  double predict(const ArrayDouble &x_t, bool use_aggregation);

  inline ulong n_features() const;
  inline ulong n_nodes() const;
  inline double step() const;

  void print() {
    for (NodeRegressor &node : nodes) {
      node.print();
    }
  }

  inline CriterionRegressor criterion() const;

  NodeRegressor &node(ulong index) {
    return nodes[index];
  }
};

/*********************************************************************************
 * OnlineForestRegressor
 *********************************************************************************/

class OnlineForestRegressor {
 private:
  // Number of Trees in the forest
  uint32_t _n_trees;
  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;
  // CriterionRegressor used for splitting (not used for now)
  CriterionRegressor _criterion;
  // Step-size used for aggregation
  double _step;
  // Number of features.
  ulong _n_features;
  // Seed for random number generation
  int _seed;
  // Verbose things or not
  bool _verbose;
  // Iteration counter
  ulong _iteration;
  // The list of trees in the forest
  std::vector<TreeRegressor> trees;
  // Random number generator for feature and threshold sampling
  Rand rand;
  // Create trees
  void create_trees();

 public:
  OnlineForestRegressor(uint32_t n_trees, double step, CriterionRegressor criterion, int32_t n_threads,
                        int seed, bool verbose);
  virtual ~OnlineForestRegressor();

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);

  inline ulong sample_feature();
  inline double sample_threshold(double left, double right);

  inline double step() const {
    return _step;
  }

  void print() {
    for (TreeRegressor &tree: trees) {
      tree.print();
    }
  }

  inline ulong n_samples() const {
    if (_iteration > 0) {
      return _iteration;
    } else {
      TICK_ERROR("You must call ``fit`` before asking for ``n_samples``.")
    }
  }

  inline ulong n_features() const {
    if (_iteration > 0) {
      return _n_features;
    } else {
      TICK_ERROR("You must call ``fit`` before asking for ``n_features``.")
    }
  }

  inline OnlineForestRegressor &set_n_features(ulong n_features) {
    if (_iteration == 0) {
      _n_features = n_features;
    } else {
      TICK_ERROR("OnlineForest::set_n_features can be called only once !")
    }
    return *this;
  }

  inline uint32_t n_trees() const {
    return _n_trees;
  }

  inline OnlineForestRegressor &set_n_trees(uint32_t n_trees) {
    _n_trees = n_trees;
    return *this;
  }

  inline int32_t n_threads() const {
    return _n_threads;
  }

  inline CriterionRegressor criterion() const {
    return _criterion;
  }

  inline int seed() const {
    return _seed;
  }

  inline OnlineForestRegressor &set_seed(int seed) {
    _seed = seed;
    rand.reseed(seed);
    return *this;
  }

  OnlineForestRegressor &set_n_threads(int32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

  inline OnlineForestRegressor &set_criterion(CriterionRegressor criterion) {
    _criterion = criterion;
    return *this;
  }


//  inline bool verbose() const;
//  inline OnlineForestRegressor &set_verbose(bool verbose);
};

#endif //TICK_ONLINEFOREST_H
