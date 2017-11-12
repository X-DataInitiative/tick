
// License: BSD 3 clause

#include "online_forest.h"

/*********************************************************************************
 * Node<NodeType> methods
 *********************************************************************************/

template<typename NodeType>
Node<NodeType>::Node(Tree<NodeType> &tree, ulong parent)
    : _tree(tree) {
  _n_samples = 0;
  _is_leaf = true;
  _left = 0;
  _right = 0;
  _weight = 1;
  _weight_tree = 1;
  this->_parent = parent;
}

template<typename NodeType>
Node<NodeType>::Node(const Node<NodeType> &node)
    : _tree(node._tree),
      _left(node._left), _right(node._right),
      _parent(node._parent),
      _feature(node._feature), _threshold(node._threshold),
      _n_samples(node._n_samples),
      _weight(node._weight), _weight_tree(node._weight_tree),
      _is_leaf(node._is_leaf),
      _x_t(node._x_t) {}

template<typename NodeType>
Node<NodeType>::Node(const Node<NodeType> &&node) : _tree(_tree) {
  _left = node._left;
  _right = node._right;
  _parent = node._parent;
  _feature = node._feature;
  _threshold = node._threshold;
  _n_samples = node._n_samples;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _x_t = node._x_t;
}

template<typename NodeType>
Node<NodeType>::~Node() {}

template<typename NodeType>
void Node<NodeType>::update_downwards(const ArrayDouble &x_t, double y_t) {
  _n_samples++;
  // TODO: Make compute loss virtual insteal
  update_weight(y_t);
  update_predict(y_t);
}

template<typename NodeType>
void Node<NodeType>::update_weight(const double y_t) {
  _weight *= exp(-step() * loss(y_t));
}

template<typename NodeType>
inline Tree<NodeType> &Node<NodeType>::tree() const {
  return _tree;
}

template<typename NodeType>
inline NodeType &Node<NodeType>::node(ulong index) const {
  return _tree.node(index);
}

template<typename NodeType>
ulong Node<NodeType>::n_features() const {
  return _tree.n_features();
}

template<typename NodeType>
inline double Node<NodeType>::step() const {
  return _tree.step();
}

template<typename NodeType>
void Node<NodeType>::print() {
  std::cout // << "Node(i: " << _index << ", p: " << _parent
      // << ", f: " << _feature
      // << ", th: " << _threshold
      << ", l: " << _left
      << ", r: " << _right
      // << ", d: " << _depth
      // << ", n: " << n_samples()
      // << ", i: " << _is_leaf
      // << ", avg: " << std::setprecision(2) << _labels_average
      // << ", feat_min=[" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
      // << _features_min[1] << "]"
      // << ", feat_max=[" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
      // << _features_max[1] << "]"
      << ")\n";
}

template<typename NodeType>
inline ulong Node<NodeType>::parent() const {
  return _parent;
}

template<typename NodeType>
inline ulong Node<NodeType>::left() const {
  return _left;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_left(ulong left) {
  _left = left;
  return *this;
}

template<typename NodeType>
inline ulong Node<NodeType>::right() const {
  return _right;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_right(ulong right) {
  _right = right;
  return *this;
}

template<typename NodeType>
inline bool Node<NodeType>::is_leaf() const {
  return _is_leaf;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_is_leaf(bool is_leaf) {
  _is_leaf = is_leaf;
  return *this;
}

template<typename NodeType>
inline ulong Node<NodeType>::feature() const {
  return _feature;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_feature(ulong feature) {
  _feature = feature;
  return *this;
}

template<typename NodeType>
inline double Node<NodeType>::threshold() const {
  return _threshold;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_threshold(double threshold) {
  _threshold = threshold;
  return *this;
}

template<typename NodeType>
inline ulong Node<NodeType>::n_samples() const {
  return _n_samples;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_n_samples(ulong n_samples) {
  _n_samples = n_samples;
  return *this;
}

template<typename NodeType>
inline double Node<NodeType>::weight() const {
  return _weight;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_weight(double weight) {
  _weight = weight;
  return *this;
}

template<typename NodeType>
inline double Node<NodeType>::weight_tree() const {
  return _weight_tree;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_weight_tree(double weight_tree) {
  _weight_tree = weight_tree;
  return *this;
}

template<typename NodeType>
inline Tree<NodeType> &Node<NodeType>::tree() const {
  return _tree;
}

template<typename NodeType>
inline ArrayDouble &Node<NodeType>::x_t() const {
  return _x_t;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_x_t(const ArrayDouble &x_t) {
  _x_t = x_t;
  return *this;
}

/*********************************************************************************
 * NodeRegressor methods
 *********************************************************************************/

NodeRegressor::NodeRegressor(Tree<NodeRegressor> &tree, ulong parent)
    : Node(tree, parent) {
  _predict = 0;
}

NodeRegressor::NodeRegressor(const NodeRegressor &node)
    : Node(node), _predict(node._predict), _y_t(node._y_t) {}

NodeRegressor::NodeRegressor(const NodeRegressor &&node)
    : Node(node) {
  _predict = node._predict;
  _y_t = node._y_t;
}

NodeRegressor::~NodeRegressor() {}

inline double NodeRegressor::predict() const {
  return _predict;
}

void NodeRegressor::update_predict(double y_t) {
  // When a node is updated, it necessarily contains already a sample
  _predict = ((_n_samples - 1) * _predict + y_t) / _n_samples;
}

double NodeRegressor::loss(const double y_t) {
  double diff = _predict - y_t;
  return diff * diff / 2;
}

void NodeRegressor::print() {
  std::cout // << "Node(idx: " << _index << ", parent: " << _parent
      // << ", f: " << _feature
      // << ", th: " << _threshold
      << ", left: " << _left
      << ", right: " << _right
      // << ", d: " << _depth
      // << ", n: " << n_samples()
      // << ", i: " << _is_leaf
      << ", thresh: " << _threshold
      << ", y_hat: " << _predict
      << ", sample: ";
  // << ", has_sample:" << _has_sample;
  if (_is_leaf) {
    std::cout << "[" << std::setprecision(2) << _x_t[0] << ", " << std::setprecision(2) << _x_t[1]
              << "]";
  } else {
    std::cout << "null";
  }
  std::cout << ", weight: " << _weight;
  std::cout << ", weight_tree: " << _weight_tree;
  std::cout << ")\n";
}

inline double NodeRegressor::y_t() const {
  return _y_t;
}

inline NodeRegressor &NodeRegressor::set_y_t(const double y_t) {
  _y_t = y_t;
  return *this;
}

/*********************************************************************************
 * Tree<NodeType> methods
 *********************************************************************************/

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &tree)
    : nodes(tree.nodes), forest(tree.forest) {
  // std::cout << "Tree::Tree(const &Tree tree)" << std::endl;
}

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &&tree) : nodes(tree.nodes), forest(tree.forest) {
}

template<typename NodeType>
ulong Node<NodeType>::n_features() const {
  return _tree.n_features();
}

template<typename NodeType>
Tree<NodeType>::Tree(OnlineForestRegressor &forest) : forest(forest) {
  // TODO: pre-allocate the vector to make things faster ?
  add_node(0, 0);
}

template<typename NodeType>
ulong Tree<NodeType>::split_leaf(ulong index, const ArrayDouble &x_t, double y_t) {
  // std::cout << "Splitting node " << index << std::endl;
  ulong left = add_node(index, iteration);
  ulong right = add_node(index, iteration);
  node(index).set_left(left).set_right(right).set_is_leaf(false);

  // TODO: better feature sampling
  ulong feature = forest.sample_feature();

  double x1_tj = x_t[feature];
  double x2_tj = node(index).sample().first[feature];
  double threshold;

  // The leaf that contains the passed sample (x_t, y_t)
  ulong data_leaf;
  ulong other_leaf;

  // std::cout << "x1_tj= " << x1_tj << " x2_tj= " << x2_tj << " threshold= " << threshold << std::endl;
  // TODO: what if x1_tj == x2_tj. Must be taken care of by sample_feature()
  if (x1_tj < x2_tj) {
    threshold = forest.sample_threshold(x1_tj, x2_tj);
    data_leaf = left;
    other_leaf = right;
  } else {
    threshold = forest.sample_threshold(x2_tj, x1_tj);
    data_leaf = right;
    other_leaf = left;
  }
  // TODO: not so sure that y_t should be passed below
  // TODO: code a move_sample

  node(index).set_feature(feature).set_threshold(threshold);

  // We pass the sample to the new leaves, and initialize the _label_average with the value
  node(data_leaf).set_sample(x_t, y_t);
  node(other_leaf).set_sample(node(index).sample());

  // Update downwards of v'
  node(other_leaf).update_down(node(index).sample().first, node(index).sample().second);
  // Update upwards of v': it's a leaf
  node(other_leaf).set_weight_tree(node(other_leaf).weight());
  // Update downwards of v''
  node(data_leaf).update_down(x_t, y_t);
  // Note: the update_up of v'' is done in the go_up method, called in fit()

  return data_leaf;
}

template<typename NodeType>
ulong Tree<NodeType>::go_downwards(const ArrayDouble &x_t, double y_t, bool predict) {
  // Find the leaf that contains the sample
  // Start at the root. Index of the root is always 0
  // If predict == true, this call to find_leaf is for
  // prediction only, so that no leaf update and splits can be done
  ulong index_current_node = 0;
  bool is_leaf = false;
  while (!is_leaf) {
    // Get the current node
    Node<NodeType> &current_node = node(index_current_node);
    if (!predict) {
      current_node.update_downwards(x_t, y_t);
    }
    // Is the node a leaf ?
    is_leaf = current_node.is_leaf();
    if (!is_leaf) {
      if (x_t[current_node.feature()] <= current_node.threshold()) {
        index_current_node = current_node.left();
      } else {
        index_current_node = current_node.right();
      }
    }
  }
  return index_current_node;
}

template<typename NodeType>
void Tree<NodeType>::go_upwards(ulong leaf_index) {
  ulong current = leaf_index;
  while (true) {
    // TODO: use a node::update_upward
    Node<NodeType> &current_node = node(current);
    if (current_node.is_leaf()) {
      current_node.set_weight_tree(current_node.weight());
    } else {
      double w = current_node.weight();
      double w0 = node(current_node.left()).weight_tree();
      double w1 = node(current_node.right()).weight_tree();
      current_node.set_weight_tree((w + w0 * w1) / 2);
//      double a = current_node.weight();
//      double b = weight_tree_left + weight_tree_right;
//      double toto;
//      if(a > b) {
//        toto = a + log(1 + exp(b - a)) - log(2);
//      } else {
//        toto = b + log(1 + exp(a - b)) - log(2);
//      }
    }
    // TODO: we must update also the root node !!!
    current = node(current).parent();
    if (current == 0) {
      break;
    }
  }
}

template<typename NodeType>
inline ulong Tree<NodeType>::n_nodes() const {
  return _n_nodes;
}

template<typename NodeType>
void Tree<NodeType>::rescale() {
  double scale = 1 / nodes[0].weight();
  for (NodeType &node : nodes) {
    node.set_weight(node.weight() * scale);
    node.set_weight_tree(node.weight_tree() * scale);
  }
}

template<typename NodeType>
void Tree<NodeType>::fit(const ArrayDouble &x_t, double y_t) {
  // TODO: Test that the size does not change within successive calls to fit
  if (iteration == 0) {
    nodes[0].set_sample(x_t, y_t);
    iteration++;
    return;
  }

  ulong leaf = go_downwards(x_t, y_t, false);
  ulong new_leaf = split_leaf(leaf, x_t, y_t);

//  for(ulong j=0; j < n_features(); ++j) {
//    double delta = std::abs(x_t[j] - node(leaf).sample().first[j]);
//    if (delta > 0.) {
//      new_leaf = split_node(leaf, x_t, y_t);
//      break;
//    }
//  }
  go_upwards(new_leaf);
  iteration++;
}

/*********************************************************************************
* TreeRegressor methods
*********************************************************************************/

TreeRegressor::TreeRegressor(OnlineForestRegressor &forest)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &tree)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &&tree)
    : Tree<NodeRegressor>(forest) {}

double TreeRegressor::predict(const ArrayDouble &x_t, bool use_aggregation) {
  ulong leaf = go_downwards(x_t, 0., true);
  if (!use_aggregation) {
    return node(leaf).y_t();
  }
  ulong current = leaf;
  // The child of the current node that does not contain the data
  ulong other;
  ulong parent;
  double weight;
  while (true) {
    NodeRegressor &current_node = node(current);
    if (current_node.is_leaf()) {
      weight = current_node.weight() * current_node.predict();
      // weight = std::exp(current_node.weight()) * current_node.labels_average();
    } else {
      weight = 0.5 * current_node.weight() * current_node.predict()
          + 0.5 * node(other).weight_tree() * weight;
//      weight = 0.5 * std::exp(current_node.weight()) * current_node.labels_average()
//          + 0.5 * std::exp(node(other).weight_tree() + weight);
    }
    parent = node(current).parent();
    if (node(parent).left() == current) {
      other = node(parent).right();
    } else {
      other = node(parent).left();
    }
    current = parent;
    // THE ROOT MUST BE INCLUDED !!!
    if (current == 0) {
      break;
    }
  }
  return weight / nodes[0].weight_tree();
  // return weight / std::exp(nodes[0].weight_tree());
}

template<typename NodeType>
ulong Tree<NodeType>::add_node(ulong parent, ulong creation_time) {
  nodes.emplace_back(*this, _n_nodes, parent, creation_time);
  return _n_nodes++;
}

template<typename NodeType>
inline ulong Tree<NodeType>::n_features() const {
  return forest.n_features();
}

template<typename NodeType>
inline double Tree<NodeType>::step() const {
  return forest.step();
}

template<typename NodeType>
inline Criterion Tree<NodeType>::criterion() const {
  return forest.criterion();
}

/*********************************************************************************
 * OnlineForestRegressor methods
 *********************************************************************************/

OnlineForestRegressor::OnlineForestRegressor(uint32_t n_trees,
                                             double step,
                                             Criterion criterion,
                                             int32_t n_threads,
                                             int seed,
                                             bool verbose)
    : trees(), _n_trees(n_trees), _step(step), _criterion(criterion), _n_threads(n_threads), _verbose(verbose) {
  // No iteration so far
  _iteration = 0;
  create_trees();
  // Seed the random number generators
  set_seed(seed);
}

OnlineForestRegressor::~OnlineForestRegressor() {}

void OnlineForestRegressor::create_trees() {
  // Just in case...
  trees.clear();
  trees.reserve(_n_trees);
  for (uint32_t i = 0; i < _n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

void OnlineForestRegressor::fit(const SArrayDouble2dPtr features,
                                const SArrayDoublePtr labels) {
  ulong n_samples = features->n_rows();
  ulong n_features = features->n_cols();
  set_n_features(n_features);
  for (ulong i = 0; i < n_samples; ++i) {
    for (TreeRegressor &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(view_row(*features, i), (*labels)[i]);
    }
    _iteration++;
  }
}

void OnlineForestRegressor::predict(const SArrayDouble2dPtr features,
                                    SArrayDoublePtr predictions,
                                    bool use_aggregation) {
  if (_iteration > 0) {
    ulong n_samples = features->n_rows();
    for (ulong i = 0; i < n_samples; ++i) {
      // The prediction is simply the average of the predictions
      double y_pred = 0;
      for (TreeRegressor &tree : trees) {
        y_pred += tree.predict(view_row(*features, i), use_aggregation);
      }
      (*predictions)[i] = y_pred / _n_trees;
    }
  } else {
    TICK_ERROR("You must call ``fit`` before ``predict``.")
  }
}

inline ulong OnlineForestRegressor::sample_feature() {
  return rand.uniform_int(0L, n_features() - 1);
}

inline double OnlineForestRegressor::sample_threshold(double left, double right) {
  return rand.uniform(left, right);
}

inline double OnlineForestRegressor::step() const {
  return _step;
}

void OnlineForestRegressor::print() {
  for (Tree<NodeRegressor> &tree: trees) {
    tree.print();
  }
}

inline ulong OnlineForestRegressor::n_samples() const {
  if (_iteration > 0) {
    return _iteration;
  } else {
    TICK_ERROR("You must call ``fit`` before asking for ``n_samples``.")
  }
}

inline ulong OnlineForestRegressor::n_features() const {
  if (_iteration > 0) {
    return _n_features;
  } else {
    TICK_ERROR("You must call ``fit`` before asking for ``n_features``.")
  }
}

inline OnlineForestRegressor &OnlineForestRegressor::set_n_features(ulong n_features) {
  if (_iteration == 0) {
    _n_features = n_features;
  } else {
    TICK_ERROR("OnlineForest::set_n_features can be called only once !")
  }
  return *this;
}

inline uint32_t OnlineForestRegressor::n_trees() const {
  return _n_trees;
}

inline OnlineForestRegressor &OnlineForestRegressor::set_n_trees(uint32_t n_trees) {
  _n_trees = n_trees;
  return *this;
}

inline int32_t OnlineForestRegressor::n_threads() const {
  return _n_threads;
}

inline OnlineForestRegressor &OnlineForestRegressor::set_n_threads(int32_t n_threads) {
  _n_threads = n_threads;
  return *this;
}

inline Criterion OnlineForestRegressor::criterion() const {
  return _criterion;
}

inline OnlineForestRegressor &OnlineForestRegressor::set_criterion(Criterion criterion) {
  _criterion = criterion;
  return *this;
}

inline int OnlineForestRegressor::seed() const {
  return _seed;
}

inline OnlineForestRegressor &OnlineForestRegressor::set_seed(int seed) {
  _seed = seed;
  rand.reseed(seed);
  return *this;
}

inline bool OnlineForestRegressor::verbose() const {
  return _verbose;
}

inline OnlineForestRegressor &OnlineForestRegressor::set_verbose(bool verbose) {
  _verbose = verbose;
  return *this;
}
