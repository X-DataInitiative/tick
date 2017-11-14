
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"



// TODO: arreter les conneries et mettre deux templates
// TODO: Node<TreeType>
// TODO: Tree<NodeType>
// TODO:
// TODO:

// TODO: faire tres attention au features binaires si le range est 0 sur toutes les coordonnées, ne rien faire
// TODO: code a classifier

// TODO: choisir la feature proportionnellement au ratio de la longueur du cote / perimetre. Ca suppose qu'on enregistre
//       les vraies dimensions de la cellule, et le threhsold est du coup aussi tiré là dedans
// TODO: choisir la feature proportionnellement au ratio des range de features, mais attention au cas de features
//       discretes
// TODO: une option pour créer une cellule vide, enfin oublier les donnes dans la cellule quand elle a ete splitee

// TODO: choix de la feature les labels

// TODO: des fit_online qui prend un mini batch et qui met à jour la foret, mais dans ce cas on ne met qu'un point par
//       cellule, du coup pas besoin d'enregistrer les sample index ou les points. Ca suppose que min_sample_split == 1

// TODO: pour la regression, on utilise la moyenne des y
// TODO: pour la classification, on utilise pas les frequences, on utilise des frequences regularisees, prior Dirichlet p_c = (n_c + 0.5) + (\sum n_c + C / 2). En fait une option

// TODO: check that not using reserve in the forest works as well...


enum class Criterion {
  unif = 0,
  mse
};


template<typename NodeType, typename ForestType>
class Tree;

/*********************************************************************************
 * Node<NodeType>
 *********************************************************************************/

template<typename TreeType>
class Node {
 protected:
  // Tree containing the node
  TreeType &_tree;
  // Index of the left child
  ulong _left;
  // Index of the right child
  ulong _right;
  // Index of the parent
  ulong _parent;
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
  // Aggregation weight for the node
  double _weight;
  // Aggregation weight for the sub-tree starting at this node
  double _weight_tree;
  // true if the node is a leaf
  bool _is_leaf;

 public:
  Node(TreeType &tree, ulong parent);
  Node(const Node&node);
  Node(const Node &&node);
  Node &operator=(const Node &) = delete;
  Node &operator=(const Node &&) = delete;
  virtual ~Node();

  // Update to apply to a node when going forward in the tree (towards leaves)
  virtual void update_downwards(const ArrayDouble &x_t, double y_t) final;
  // Update to apply to a node when going upward in the tree (towards the root)
  virtual void update_upwards() final;
  // Update of the aggregation weights
  virtual void update_weight(double y_t);
  // Update the prediction of the label
  virtual void update_predict(double y_t) = 0;
  // Loss function used for aggregation
  virtual double loss(const double y_t) = 0;

  inline TreeType &tree() const;
  inline Node &node(ulong index) const;
  ulong n_features() const;
  inline double step() const;

  virtual void print();

  inline ulong parent() const;
  inline ulong left() const;
  inline Node &set_left(ulong left);
  inline ulong right() const;
  inline Node &set_right(ulong right);
  inline bool is_leaf() const;
  inline Node &set_is_leaf(bool is_leaf);
  inline ulong feature() const;
  inline Node &set_feature(ulong feature);
  inline double threshold() const;
  inline Node &set_threshold(double threshold);
  inline ulong n_samples() const;
  inline Node &set_n_samples(ulong n_samples);
  inline double weight() const;
  inline Node &set_weight(double weight);
  inline double weight_tree() const;
  inline Node &set_weight_tree(double weight);
  inline const ArrayDouble &x_t() const;
  inline double x_t(const ulong j) const;
  inline Node &set_x_t(const ArrayDouble &x_t);
  inline double y_t() const;
  inline Node& set_y_t(const double y_t);
};


/*********************************************************************************
 * NodeRegressor
 *********************************************************************************/

class TreeRegressor;

class NodeRegressor : public Node<TreeRegressor> {
  // using node_type = NodeRegressor;
 private:
  // Average of the labels in the node (regression only for now)
  double _predict = 0;
  // Label of the stored sample point
  double _y_t;

 public:
  NodeRegressor(TreeRegressor &tree, ulong parent);
  NodeRegressor(const NodeRegressor &node);
  NodeRegressor(const NodeRegressor &&node);
  NodeRegressor &operator=(const NodeRegressor &) = delete;
  NodeRegressor &operator=(const NodeRegressor &&) = delete;
  virtual ~NodeRegressor();

  inline double predict() const;
  void update_weight(double y_t);
  virtual void update_predict(double y_t);
  virtual double loss(const double y_t);
  virtual void print();
};

/*********************************************************************************
 * NodeClassifier
 *********************************************************************************/

class TreeClassifier;

class NodeClassifier : public Node<TreeClassifier> {
 private:
  // Score of each class
  ArrayDouble _predict;
  // Label of the stored sample point
  double _y_t;

 public:
  NodeClassifier(TreeClassifier &tree, ulong parent);
  NodeClassifier(const NodeClassifier &node);
  NodeClassifier(const NodeClassifier &&node);
  NodeClassifier &operator=(const NodeClassifier &) = delete;
  NodeClassifier &operator=(const NodeClassifier &&) = delete;
  virtual ~NodeClassifier();

  inline const ArrayDouble& predict() const;
  // virtual void update_weight(double y_t);
  virtual void update_predict(double y_t);
  virtual double loss(const double y_t);
  virtual void print();

  inline uint8_t n_classes() const;
};

template <typename NodeType, typename TreeType>
class OnlineForest;


/*********************************************************************************
 * Tree<NodeType>
 *********************************************************************************/

template<typename NodeType, typename ForestType>
class Tree {
 protected:
  // The forest of the tree
  ForestType &_forest;
  // Number of nodes in the tree
  ulong _n_nodes = 0;
  // Iteration counter
  ulong iteration = 0;
  // Nodes of the tree
  std::vector<NodeType> nodes = std::vector<NodeType>();
  // Split the node at given index
  ulong split_leaf(ulong index, const ArrayDouble &x_t, double y_t);
  // Add nodes in the tree
  virtual ulong add_node(ulong parent, ulong creation_time) = 0;

  ulong go_downwards(const ArrayDouble &x_t, double y_t, bool predict, ulong &depth);
  void go_upwards(ulong leaf_index);

 public:
  Tree(ForestType &forest);
  Tree(const Tree<NodeType, ForestType> &tree);
  Tree(const Tree<NodeType, ForestType> &&tree);
  Tree &operator=(const Tree<NodeType, ForestType> &) = delete;
  Tree &operator=(const Tree<NodeType, ForestType> &&) = delete;
  ~Tree() {}

  void fit(const ArrayDouble &x_t, double y_t);

  inline ulong n_features() const;
  inline ulong n_nodes() const;
  inline double step() const;

  void print() {
    for (NodeType &node : nodes) {
      node.print();
    }
  }

  inline Criterion criterion() const;

  NodeType &node(ulong index) {
    return nodes[index];
  }
};

/*********************************************************************************
 * TreeRegressor
 *********************************************************************************/

class OnlineForestRegressor;

class TreeRegressor : public Tree<NodeRegressor, OnlineForestRegressor> {
 public:
  TreeRegressor(OnlineForestRegressor &forest);
  TreeRegressor(const TreeRegressor &tree);
  TreeRegressor(const TreeRegressor &&tree);
  TreeRegressor &operator=(const TreeRegressor &) = delete;
  TreeRegressor &operator=(const TreeRegressor &&) = delete;

  virtual ulong add_node(ulong parent, ulong creation_time);
  double predict(const ArrayDouble &x_t, bool use_aggregation);
};

/*********************************************************************************
 * TreeClassifier
 *********************************************************************************/

class OnlineForestClassifier;

class TreeClassifier : public Tree<NodeClassifier, OnlineForestClassifier> {
 public:
  TreeClassifier(OnlineForestClassifier &forest);
  TreeClassifier(const TreeClassifier &tree);
  TreeClassifier(const TreeClassifier &&tree);
  TreeClassifier &operator=(const TreeClassifier &) = delete;
  TreeClassifier &operator=(const TreeClassifier &&) = delete;

  virtual ulong add_node(ulong parent, ulong creation_time);
  double predict(const ArrayDouble &x_t, bool use_aggregation);

  inline uint8_t n_classes() const;
};


/*********************************************************************************
 * OnlineForest
 *********************************************************************************/

template <typename NodeType, typename TreeType>
class OnlineForest {
 protected:
  // Number of Trees in the forest
  uint32_t _n_trees;
  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;
  // Criterion used for splitting (not used for now)
  Criterion _criterion;
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
  std::vector<TreeType> trees;
  // Random number generator for feature and threshold sampling
  Rand rand;
  // Create trees
  virtual void create_trees() = 0;

 public:
  OnlineForest(uint32_t n_trees, double step, Criterion criterion, int32_t n_threads, int seed, bool verbose);
  virtual ~OnlineForest();

  virtual void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels) final;

  virtual void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation) = 0;

  void clear() {
    _iteration = 0;
    create_trees();
  }

  inline ulong sample_feature();
  inline double sample_threshold(double left, double right);

  inline double step() const {
    return _step;
  }

  void print() {
    for (TreeType &tree: trees) {
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

 private:
  inline OnlineForest &set_n_features(ulong n_features) {
    _n_features = n_features;
    return *this;
  }

 public:
  inline uint32_t n_trees() const {
    return _n_trees;
  }

  inline OnlineForest &set_n_trees(uint32_t n_trees) {
    _n_trees = n_trees;
    return *this;
  }

  inline int32_t n_threads() const {
    return _n_threads;
  }

  inline OnlineForest &set_n_threads(int32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

  inline Criterion criterion() const {
    return _criterion;
  }

  inline OnlineForest &set_criterion(Criterion criterion) {
    _criterion = criterion;
    return *this;
  }

  inline int seed() const {
    return _seed;
  }

  inline OnlineForest &set_seed(int seed) {
    _seed = seed;
    rand.reseed(seed);
    return *this;
  }
//  inline bool verbose() const;
//  inline OnlineForestRegressor &set_verbose(bool verbose);
};


/*********************************************************************************
 * OnlineForestRegressor
 *********************************************************************************/

class OnlineForestRegressor : public OnlineForest<NodeRegressor, TreeRegressor> {
 public:
  OnlineForestRegressor(uint32_t n_trees, double step, Criterion criterion, int32_t n_threads, int seed, bool verbose);
  virtual ~OnlineForestRegressor();

  virtual void create_trees() final;

  virtual void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);
};

/*********************************************************************************
 * OnlineForestClassifier
 *********************************************************************************/

class OnlineForestClassifier : public OnlineForest<NodeClassifier, TreeClassifier> {
 public:
  OnlineForestClassifier(uint32_t n_trees, double step, Criterion criterion, int32_t n_threads, int seed, bool verbose);
  virtual ~OnlineForestClassifier();

  virtual void create_trees() final;

  virtual void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);

  inline ulong n_classes() const {
    return 0.;
  }
};


#endif //TICK_ONLINEFOREST_H
