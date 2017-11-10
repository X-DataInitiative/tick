
#ifndef TICK_ONLINEFOREST_H
#define TICK_ONLINEFOREST_H

// License: BSD 3 clause

#include "base.h"
#include <iomanip>
#include "../../random/src/rand.h"



// TODO: redo everything with only samples passed to the tree and not indexes
// TODO: one _sample per node

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
  Tree<NodeType> &_tree;
  // Index in the list of nodes of the tree (to be removed later)
  ulong _index;
  // Index of the left child
  ulong _left;
  // Index of the right child
  ulong _right;
  // Index of the parent
  ulong _parent;
  // Creation time of the node (iteration index)
  ulong _time;
  // Index of the feature used for the split
  ulong _feature;
  // Threshold used for the split
  double _threshold;
  // Depth of the node
  // ulong _depth = 0;
  // Number of samples in the node
  ulong _n_samples;

  // The sample saved in the node for splitting using its range
  std::pair<ArrayDouble, double> _sample;
  // ArrayDouble _sample;

  // Aggregation weight for the node
  double _weight;
  // Aggregation weight for the sub-tree starting at this node
  double _weight_tree;

  // Minimum value of each feature (minimum range)
  // ArrayDouble _features_min;
  // Maximum value of each feature (maximum range)
  // ArrayDouble _features_max;
  // The indexes (row numbers) of the samples currently in the node
  // std::vector<ulong> _samples;
  // true if the node is a leaf
  bool _is_leaf;

  // True whenever the node contains a sample
  // bool _has_sample;

 public:
  Node(Tree<NodeType> &tree, ulong index, ulong parent, ulong creation_time);
  Node(const Node<NodeType> &node);
  Node(const Node<NodeType> &&node);
  Node<NodeType> &operator=(const Node<NodeType> &) = delete;
  Node<NodeType> &operator=(const Node<NodeType> &&) = delete;
  virtual ~Node();

  // Update to apply to a node when going forward in the tree (towards leaves)
  virtual void update_down(const ArrayDouble &x_t, double y_t);

  // Update the range of the node
  // void update_range(ulong sample_index);

  virtual void update_weight(const ArrayDouble &x_t, double y_t) = 0;

  // Update the statistics about the labels of the points in this node
  virtual void update_label_stats(double y_t) = 0;


  inline Tree<NodeType> &tree() const;
  ulong n_features() const;

  inline ulong index() const;
  inline ulong parent() const;
  inline ulong left() const;
  inline Node<NodeType> &set_left(ulong left);
  inline ulong right() const;
  inline Node<NodeType> &set_right(ulong right);
  inline const bool is_leaf() const;
  inline Node<NodeType> &set_is_leaf(bool is_leaf);
  inline const bool has_sample() const;
  inline Node<NodeType> & set_has_sample(bool has_sample);

  inline ulong time() const;

  inline ulong feature() const;
  inline Node<NodeType> &set_feature(ulong feature);
  inline double threshold() const;
  inline Node<NodeType> &set_threshold(double threshold);
  inline ulong n_samples() const;
  inline Node<NodeType> &set_n_samples(ulong n_samples);

  // inline ulong depth() const;
  // inline Node<NodeType> &set_depth(ulong depth);

  inline double weight() const;
  inline Node<NodeType> & set_weight(double weight);
  inline double weight_tree() const;
  inline Node<NodeType> & set_weight_tree(double weight);


  inline std::pair<ArrayDouble, double>& sample();

  virtual Node<NodeType> &set_sample(const ArrayDouble& x_t, double y_t) = 0;

  virtual Node<NodeType> &set_sample(const std::pair<ArrayDouble, double> & sample) = 0;

//  inline const ArrayDouble &features_min() const;
//  inline Node<NodeType> &set_features_min(const ArrayDouble &features_min);
//  inline Node<NodeType> &set_features_min(const ulong j, const double x);
//  inline const ArrayDouble &features_max() const;
//  inline Node<NodeType> &set_features_max(const ulong j, const double x);
//  inline Node<NodeType> &set_features_max(const ArrayDouble &features_max);
  // inline const std::vector<ulong> &samples() const;
  // inline ulong sample(ulong index) const;
  // inline Node<NodeType> &add_sample(ulong index);

//  inline ArrayDouble get_features(ulong sample_index) const;
//  inline double get_label(ulong sample_index) const;
  virtual void print();

};


class NodeRegressor : public Node<NodeRegressor> {
 private:
  // Average of the labels in the node (regression only for now)
  double _labels_average = 0;

 public:
  NodeRegressor(Tree<NodeRegressor> &tree, ulong index, ulong parent, ulong creation_time);
  NodeRegressor(const NodeRegressor &node);
  NodeRegressor(const NodeRegressor &&node);
  virtual ~NodeRegressor();

  inline double labels_average() const;
  inline NodeRegressor&set_labels_average(double avg);

  virtual void update_label_stats(double y_t);

  virtual void update_weight(const ArrayDouble &x_t, double y_t);

  virtual NodeRegressor &set_sample(const ArrayDouble& x_t, double y_t);
  virtual NodeRegressor &set_sample(const std::pair<ArrayDouble, double> & sample);

  virtual void print();
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
  ulong split_leaf(ulong index, const ArrayDouble &x_t, double y_t);

  virtual ulong add_node(ulong parent, ulong creation_time);

  ulong go_down(const ArrayDouble &x_t, double y_t, bool predict);

  void go_up(ulong leaf_index);

  std::pair<ulong, double> sample_feature_and_threshold(ulong index);

 public:
  Tree(OnlineForestRegressor &forest);
  Tree(const Tree<NodeType> &tree);
  Tree(const Tree<NodeType> &&tree);
  Tree &operator=(const Tree<NodeType> &) = delete;
  Tree &operator=(const Tree<NodeType> &&) = delete;

  void fit(const ArrayDouble& x_t, double y_t);

  // double predict(const ArrayDouble& x_t);

  inline ulong n_features() const;

  inline ulong n_nodes() const {
    return _n_nodes;
  }

//  inline ulong depth() const {
//    return _depth;
//  }

  void print() {
    // std::cout << "start print" << std::endl;
    for(NodeType& node : nodes) {
      node.print();
    }
    // std::cout << "end print" << std::endl;
  }

//  inline uint32_t min_samples_split() const;
  inline Criterion criterion() const;
//  inline ArrayDouble get_features(ulong sample_index) const;
//  inline ArrayDouble get_features_predict(ulong sample_index) const;
//  inline double get_label(ulong sample_index) const;

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

  double predict(const ArrayDouble& x_t, bool use_aggregation);
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
  // uint32_t _min_samples_split;

  // Number of candidate splits to be considered
  // uint32_t _n_splits;

  Criterion _criterion;
  ulong _n_features;
  bool _n_features_known;

  // ulong _iteration;

  // int32_t _max_depth;

  int _seed;
  bool _verbose;
  bool _warm_start;

  // Iteration counter
  ulong _iteration;

  // The matrix of features used for fitting
//  SArrayDouble2dPtr _features_fit;
//  // The vector of labels used for fitting
//  SArrayDoublePtr _labels_fit;
//  // The vector of features used for prediction
//  SArrayDouble2dPtr _features_predict;

  // The list of trees in the forest
  std::vector<TreeRegressor> trees;

  // Do the forest received data
  // bool _fitted;

  // ulong n_features;
  // Type of random sampling
  // CycleType cycle_type;
  // An array that allows to store the sampled random permutation
  // ArrayULong permutation;
  // Current index in the permutation (useful when using random permutation sampling)
  // ulong i_perm;
  // A flag that specify if random permutation is ready to be used or not
  // bool permutation_ready;
  // Init permutation array in case of Random is srt to permutation
  // void init_permutation();

  // Random number generator for feature and threshold sampling
  Rand rand;

  // ulong get_next_sample();
  // void shuffle();

 public:
  OnlineForestRegressor(uint32_t n_trees,
                        Criterion criterion,
                        // int32_t max_depth,
                        // uint32_t min_samples_split,
                        int32_t n_threads,
                        int seed,
                        bool verbose
                        // bool warm_start,
                        // uint32_t n_splits
  );

  ~OnlineForestRegressor() {}

  void create_trees();

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);

  void predict(const SArrayDouble2dPtr features, SArrayDoublePtr predictions, bool use_aggregation);

  inline ulong n_features() const {
    if (_iteration > 0) {
      return _n_features;
    } else {
      TICK_ERROR("OnlineForest::n_features: the forest has no data yet.")
    }
  }

  inline OnlineForestRegressor& set_n_features(ulong n_features) {
    if (_iteration == 0) {
      _n_features = n_features;
    } else {
      TICK_ERROR("OnlineForest::set_n_features can be called only once !")
    }
    return *this;
  }

  inline ulong n_samples() const {
    if (_iteration > 0) {
      return _iteration;
    } else {
      TICK_ERROR("OnlineForest::n_samples the forest has no data yet.")
    }
  }

//  inline ArrayDouble features(ulong sample_index) const {
//    return view_row(*_features_fit, sample_index);
//  }
//
//  inline ArrayDouble features_predict(ulong sample_index) const {
//    return view_row(*_features_predict, sample_index);
//  }
//
//  inline double label(ulong i) const {
//    return (*_labels_fit)[i];
//  }

  void print() {
    // std::cout << "Forest" << std::endl;
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

//  inline uint32_t n_splits() const {
//    return _n_splits;
//  }
//
//  inline OnlineForestRegressor &set_n_splits(uint32_t n_splits) {
//    _n_splits = n_splits;
//    return *this;
//  }

  inline int32_t n_threads() const {
    return _n_threads;
  }

  inline OnlineForestRegressor &set_n_threads(int32_t n_threads) {
    _n_threads = n_threads;
    return *this;
  }

//  inline uint32_t min_samples_split() const {
//    return _min_samples_split;
//  }
//
//  inline OnlineForestRegressor &set_min_samples_split(uint32_t min_samples_split) {
//    _min_samples_split = min_samples_split;
//    return *this;
//  }

  inline Criterion criterion() const {
    return _criterion;
  }

  inline OnlineForestRegressor &set_criterion(Criterion criterion) {
    _criterion = criterion;
    return *this;
  }

//  inline int32_t max_depth() const {
//    return _max_depth;
//  }
//
//  inline OnlineForestRegressor &set_max_depth(int32_t max_depth) {
//    _max_depth = max_depth;
//    return *this;
//  }

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
