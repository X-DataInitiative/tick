
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"


// TODO: faire tres attention au features binaires si le range est 0 sur toutes les coordonnées, ne rien faire
// TODO: compute the depth of the tree and enable thd max_depth option
// TODO: code the least-squares criterion
// TODO: remove set_data and code only a fit method
// TODO: code a classifier
// TODO: warm-start option

// TODO: choisir la feature proportionnellement au ratio de la longueur du cote / perimetre. Ca suppose qu'on enregistre
//       les vraies dimensions de la cellule, et le threhsold est du coup aussi tiré là dedans
// TODO: choisir la feature proportionnellement au ratio des range de features, mais attention au cas de features
//       discretes
// TODO: une option pour créer une cellule vide, enfin oublier les donnes dans la cellule quand elle a ete splitee

// TODO: choix de la feature les labels

// TODO: des fit_online qui prend un mini batch et qui met à jour la foret, mais dans ce cas on ne met qu'un point par
//       cellule, du coup pas besoin d'enregistrer les sample index ou les points. Ca suppose que min_sample_split == 1

// TODO: dans le cas min_sample_spit == 1, ne pas couper jusque a chaque arrivee de point, mais quand le range change
//       (cas des features discretes ou binaires...), si il est egale, on met a jour la moyenne, ou les comptages de labels, et le nombre de sample

// TODO: pour la regression, on utilise la moyenne des y
// TODO: pour la classification, on utilise pas les frequences, on utilise des frequences regularisees, prior Dirichlet p_c = (n_c + 0.5) + (\sum n_c + C / 2). En fait une option

// TODO: check that not using reserve in the forest works as well...


enum class Criterion {
  unif = 0,
  mse
};

// Forward declaration of a Tree
template <typename NodeType>
class Tree;

template <typename NodeType>
class Node {
 protected:
  // The tree of the node
  Tree<NodeType> &in_tree;

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
  // Depth of the node
  ulong _depth = 0;
  // Impurity of the node
  double _impurity = 0;
  // Number of samples in the node
  ulong _n_samples;

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
  Node(Tree<NodeType> &tree, ulong index, ulong parent, ulong creation_time);

  Node(const Node<NodeType> &node);

  Node(const Node<NodeType> &&node);

  virtual ~Node() {
    // std::cout << "~Node()\n";
  }

  Node<NodeType> &operator=(const Node<NodeType> &) = delete;
  Node<NodeType> &operator=(const Node<NodeType> &&) = delete;

  inline ulong index() const {
    return _index;
  }

  inline ulong left() const {
    return _left;
  }

  inline Node<NodeType> &set_left(ulong left) {
    _left = left;
    return *this;
  }

  inline ulong right() const {
    return _right;
  }

  inline Node<NodeType> &set_right(ulong right) {
    _right = right;
    return *this;
  }

  inline const bool is_leaf() const {
    return _is_leaf;
  }

  inline Node<NodeType> &set_is_leaf(bool is_leaf) {
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

  inline Node<NodeType> &set_feature(ulong feature) {
    _feature = feature;
    return *this;
  }

  inline double threshold() const {
    return _threshold;
  }

  inline Node<NodeType> &set_threshold(double threshold) {
    _threshold = threshold;
    return *this;
  }

  inline double impurity() const {
    return _impurity;
  }

  inline ulong n_samples() const {
    return _n_samples;
  }

  inline Node<NodeType> &set_n_samples(ulong n_samples) {
    _n_samples = n_samples;
    return *this;
  }

  inline ulong depth() const {
    return _depth;
  }

  inline Node<NodeType> &set_depth(ulong depth) {
    _depth = depth;
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

  inline Node<NodeType> &set_features_min(const ArrayDouble &features_min) {
    _features_min = features_min;
    return *this;
  }

  inline Node<NodeType> &set_features_min(const ulong j, const double x) {
    _features_min[j] = x;
    return *this;
  }

  inline const ArrayDouble &features_max() const {
    return _features_max;
  }

  inline Node<NodeType> &set_features_max(const ulong j, const double x) {
    _features_max[j] = x;
    return *this;
  }

  inline Node<NodeType> &set_features_max(const ArrayDouble &features_max) {
    _features_max = features_max;
    return *this;
  }

  inline const std::vector<ulong> &samples() const {
    return _samples;
  }

  inline ulong sample(ulong index) const {
    return _samples[index];
  }

  inline Node<NodeType> &add_sample(ulong index) {
    _samples.push_back(index);
    return *this;
  }

  inline Tree<NodeType> &get_tree() const {
    return in_tree;
  }

  // Update the statistics of the node using the sample
  virtual void update(ulong sample_index, bool do_update_range = true);

  void update_range(ulong sample_index);

  virtual void update_label_stats(ulong sample_index) = 0;

  inline ArrayDouble get_features(ulong sample_index) const;

  ulong n_features() const;

  inline double get_label(ulong sample_index) const;

  void print() {
    std::cout << "Node(i: " << _index << ", p: " << _parent
              // << ", f: " << _feature
              // << ", th: " << _threshold
              << ", l: " << _left
              << ", r: " << _right
              << ", d: " << _depth
              // << ", n: " << n_samples()
              // << ", i: " << _is_leaf
              << ", t: " << _creation_time
              // << ", avg: " << std::setprecision(2) << _labels_average
              // << ", feat_min=[" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
              // << _features_min[1] << "]"
              // << ", feat_max=[" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
              // << _features_max[1] << "]"
              << ")\n";
  }
};


class NodeRegressor : public Node<NodeRegressor> {
 private:
  // Average of the labels in the node (regression only for now)
  double _labels_average = 0;

 public:
  NodeRegressor(Tree<NodeRegressor> &tree, ulong index, ulong parent, ulong creation_time);

  NodeRegressor(const NodeRegressor &node);

  NodeRegressor(const NodeRegressor &&node);

  virtual ~NodeRegressor() {
    // std::cout << "~Node()\n";
  }

  NodeRegressor &operator=(const NodeRegressor &) = delete;
  NodeRegressor &operator=(const NodeRegressor &&) = delete;

  inline double labels_average() const {
    return _labels_average;
  }

  inline Node &set_labels_average(double avg) {
    _labels_average = avg;
    return *this;
  }

  virtual void update_label_stats(ulong sample_index);
};


class OnlineForestRegressor;


template <typename NodeType>
class Tree {
  // friend class NodeType;

 protected:
  std::vector<NodeType> nodes = std::vector<NodeType>();

  bool already_fitted = false;

  // Number of nodes in the tree
  ulong _n_nodes = 0;

  // Depth of the tree
  ulong _depth = 0;

  // The forest of the tree
  OnlineForestRegressor &forest;

  // Iteration counter
  ulong iteration = 0;

  // Split the node at given index
  void split_node(ulong index);

  virtual ulong add_node(ulong parent, ulong creation_time);

  ulong find_leaf(ulong sample_index, bool predict);

  std::pair<ulong, double> sample_feature_and_threshold(ulong index);

 public:
  Tree(OnlineForestRegressor &forest);
  Tree(const Tree<NodeType> &tree);
  Tree(const Tree<NodeType> &&tree);
  Tree &operator=(const Tree<NodeType> &) = delete;
  Tree &operator=(const Tree<NodeType> &&) = delete;

  // Launch a pass on the given data
  void fit(ulong sample_index);

  double predict(ulong sample_index);

  inline ulong n_features() const;

  inline ulong n_nodes() const {
    return _n_nodes;
  }

  inline ulong depth() const {
    return _depth;
  }

  void print() {
    for(NodeType& node : nodes) {
      node.print();
    }
  }

  inline uint32_t min_samples_split() const;
  inline Criterion criterion() const;
  inline ArrayDouble get_features(ulong sample_index) const;
  inline ArrayDouble get_features_predict(ulong sample_index) const;
  inline double get_label(ulong sample_index) const;

  NodeType& node(ulong index) {
    return nodes[index];
  }

  ~Tree() {}
};


class TreeRegressor : public Tree<NodeRegressor> {
 public:
  TreeRegressor(OnlineForestRegressor &forest);
  TreeRegressor(const TreeRegressor &tree);
  TreeRegressor(const TreeRegressor &&tree);
  TreeRegressor &operator=(const TreeRegressor &) = delete;
  TreeRegressor &operator=(const TreeRegressor &&) = delete;
};



// Type of randomness used when sampling at random data points
enum class CycleType {
  uniform = 0,
  permutation,
  sequential
};

class OnlineForestRegressor {

 private:
  // Number of Trees in the forest
  uint32_t _n_trees;

  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;

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
  std::vector<Tree<NodeRegressor>> trees;

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

  // Random number generator for feature and threshold sampling
  Rand rand;

  ulong get_next_sample();

  void shuffle();

 public:
  OnlineForestRegressor(uint32_t n_trees,
                        Criterion criterion,
                        int32_t max_depth,
                        uint32_t min_samples_split,
                        int32_t n_threads,
                        int seed,
                        bool verbose,
                        bool warm_start,
                        uint32_t n_splits);

  ~OnlineForestRegressor() {
    // std::cout << "~OnlineForest()\n";
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
    for (Tree<NodeRegressor> &tree: trees) {
      tree.print();
    }
  }

  inline uint32_t n_trees() const {
    return _n_trees;
  }

  inline OnlineForestRegressor &set_n_trees(uint32_t n_trees) {
    _n_trees = n_trees;
    return *this;
  }

  inline uint32_t n_splits() const {
    return _n_splits;
  }

  inline OnlineForestRegressor &set_n_splits(uint32_t n_splits) {
    _n_splits = n_splits;
    return *this;
  }

  inline int32_t n_threads() const {
    return _n_threads;
  }

  inline OnlineForestRegressor &set_n_threads(int32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

  inline uint32_t min_samples_split() const {
    return _min_samples_split;
  }

  inline OnlineForestRegressor &set_min_samples_split(uint32_t min_samples_split) {
    _min_samples_split = min_samples_split;
    return *this;
  }

  inline Criterion criterion() const {
    return _criterion;
  }

  inline OnlineForestRegressor &set_criterion(Criterion criterion) {
    _criterion = criterion;
    return *this;
  }

  inline int32_t max_depth() const {
    return _max_depth;
  }

  inline OnlineForestRegressor &set_max_depth(int32_t max_depth) {
    _max_depth = max_depth;
    return *this;
  }

  inline int seed() const {
    return _seed;
  }

  inline OnlineForestRegressor &set_seed(int seed) {
    _seed = seed;
    rand.reseed(seed);
    return *this;
  }

  inline bool verbose() const {
    return _verbose;
  }

  inline OnlineForestRegressor &set_verbose(bool verbose) {
    _verbose = verbose;
    return *this;
  }

  inline bool warm_start() const {
    return _warm_start;
  }

  inline OnlineForestRegressor &set_warm_start(bool warm_start) {
    _warm_start = warm_start;
    return *this;
  }

  inline ulong sample_feature() {
    return rand.uniform_int(0L, n_features() - 1);
  }

  inline double sample_threshold(double left, double right) {
    return rand.uniform(left, right);
  }

};

#endif //TICK_ONLINEFOREST_H
