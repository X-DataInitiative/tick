
// License: BSD 3 clause

#include "online_forest.h"

/*********************************************************************************
 * Node<NodeType> methods
 *********************************************************************************/

Node::Node(Tree &tree, ulong parent)
    : _tree(tree) {
  _n_samples = 0;
  _is_leaf = true;
  _left = 0;
  _right = 0;
  // _weight = 1;
  // _weight_tree = 1;
  _weight = 0;
  _weight_tree = 0;
  this->_parent = parent;
}

Node::Node(const Node &node)
    : _tree(node._tree),
      _left(node._left), _right(node._right), _parent(node._parent),
      _feature(node._feature), _threshold(node._threshold),
      _n_samples(node._n_samples),
      _x_t(node._x_t),
      _y_t(node._y_t),
      _weight(node._weight), _weight_tree(node._weight_tree),
      _is_leaf(node._is_leaf) {}

Node::Node(const Node &&node) : _tree(node._tree) {
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


Node::~Node() {}

void Node::update_downwards(const ArrayDouble &x_t, double y_t) {
  _n_samples++;
  // TODO: Make compute loss virtual insteal
  update_weight(y_t);
  update_predict(y_t);
}

void Node::update_upwards() {
  if (_is_leaf) {
    _weight_tree = _weight;
  } else {
    double a = _weight;
    double b = node(_left).weight_tree() + node(_right).weight_tree();
    if (a > b) {
      _weight_tree = a + log((1 + exp(b - a)) / 2);
    } else {
      _weight_tree = b + log((1 + exp(a - b)) / 2);
    }
    // _weight_tree = (_weight + node(_left).weight_tree() * node(_right).weight_tree()) / 2;
  }
}

void Node::update_weight(const double y_t) {
  // _weight *= exp(-step() * loss(y_t));
  _weight -= step() * loss(y_t);
}

inline Tree &Node::tree() const {
  return _tree;
}

inline Node &Node::node(ulong index) const {
  return _tree.node(index);
}

ulong Node::n_features() const {
  return _tree.n_features();
}

inline double Node::step() const {
  return _tree.step();
}

void Node::print() {
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

inline ulong Node::parent() const {
  return _parent;
}

inline ulong Node::left() const {
  return _left;
}

inline Node &Node::set_left(ulong left) {
  _left = left;
  return *this;
}

inline ulong Node::right() const {
  return _right;
}

inline Node &Node::set_right(ulong right) {
  _right = right;
  return *this;
}

inline bool Node::is_leaf() const {
  return _is_leaf;
}

inline Node &Node::set_is_leaf(bool is_leaf) {
  _is_leaf = is_leaf;
  return *this;
}

inline ulong Node::feature() const {
  return _feature;
}

inline Node &Node::set_feature(ulong feature) {
  _feature = feature;
  return *this;
}

inline double Node::threshold() const {
  return _threshold;
}

inline Node &Node::set_threshold(double threshold) {
  _threshold = threshold;
  return *this;
}

inline ulong Node::n_samples() const {
  return _n_samples;
}

inline Node &Node::set_n_samples(ulong n_samples) {
  _n_samples = n_samples;
  return *this;
}

inline double Node::weight() const {
  return _weight;
}

inline Node &Node::set_weight(double weight) {
  _weight = weight;
  return *this;
}

inline double Node::weight_tree() const {
  return _weight_tree;
}

inline Node &Node::set_weight_tree(double weight_tree) {
  _weight_tree = weight_tree;
  return *this;
}

inline const ArrayDouble &Node::x_t() const {
  return _x_t;
}

inline double Node::x_t(const ulong j) const {
  return _x_t[j];
}

inline Node &Node::set_x_t(const ArrayDouble &x_t) {
  _x_t = x_t;
  return *this;
}

inline double Node::y_t() const {
  return _y_t;
}

inline Node &Node::set_y_t(const double y_t) {
  _y_t = y_t;
  return *this;
}

/*********************************************************************************
 * NodeRegressor methods
 *********************************************************************************/

NodeRegressor::NodeRegressor(Tree &tree, ulong parent)
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

/*********************************************************************************
 * NodeClassifier methods
 *********************************************************************************/

NodeClassifier::NodeClassifier(Tree &tree, ulong parent)
    : Node(tree, parent) {
  _predict = ArrayDouble(n_classes());
  _predict.fill(static_cast<double>(1) / n_classes());

}

NodeClassifier::NodeClassifier(const NodeClassifier &node)
    : Node(node), _predict(node._predict), _y_t(node._y_t) {}

NodeClassifier::NodeClassifier(const NodeClassifier &&node)
    : Node(node) {
  _predict = node._predict;
  _y_t = node._y_t;
}

NodeClassifier::~NodeClassifier() {}

inline const ArrayDouble &NodeClassifier::predict() const {
  return _predict;
}

void NodeClassifier::update_predict(double y_t) {
  // When a node is updated, it necessarily contains already a sample
  // _predict = ((_n_samples - 1) * _predict + y_t) / _n_samples;
}

double NodeClassifier::loss(const double y_t) {
  // double diff = _predict - y_t;
  // return diff * diff / 2;
  return 0;
}

void NodeClassifier::print() {
  std::cout << ", parent: " << _parent
      // << ", f: " << _feature
      // << ", th: " << _threshold
      << ", left: " << _left
      << ", right: " << _right
      // << ", d: " << _depth
      // << ", n: " << n_samples()
      // << ", i: " << _is_leaf
      << ", thresh: " << _threshold
      // << ", y_hat: " << _predict
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

inline uint8_t NodeClassifier::n_classes() const {
  return _tree.n_classes();
}

/*********************************************************************************
 * Tree<NodeType> methods
 *********************************************************************************/

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &tree)
    : nodes(tree.nodes), forest(tree.forest) {
}

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &&tree) : nodes(tree.nodes), forest(tree.forest) {
}

template<typename NodeType>
Tree<NodeType>::Tree(OnlineForest &forest) : forest(forest) {
  // TODO: pre-allocate the vector to make things faster ?
  add_node(0, 0);
}

template<typename NodeType>
ulong Tree<NodeType>::split_leaf(ulong index, const ArrayDouble &x_t, double y_t) {
  // Add the leaf nodes in the tree
  ulong left = add_node(index, iteration);
  ulong right = add_node(index, iteration);
  // Give information to the splitted node about its childs
  NodeType &current_node = node(index);
  node(index).set_left(left).set_right(right).set_is_leaf(false);

  // TODO: better feature sampling
  ulong feature = forest.sample_feature();

  double x_tj = x_t[feature];
  double x2_tj = current_node.x_t(feature);
  double threshold;
  // The leaf that contains the passed sample (x_t, y_t)
  ulong data_leaf;
  // The other leaf
  ulong other_leaf;
  // TODO: what if x1_tj == x2_tj. Must be taken care of by sample_feature()
  if (x_tj < x2_tj) {
    threshold = forest.sample_threshold(x_tj, x2_tj);
    data_leaf = left;
    other_leaf = right;
  } else {
    threshold = forest.sample_threshold(x2_tj, x_tj);
    data_leaf = right;
    other_leaf = left;
  }
  // TODO: code a move_sample
  current_node.set_feature(feature).set_threshold(threshold);
  NodeType &data_leaf_node = node(data_leaf);
  NodeType &other_leaf_node = node(other_leaf);
  // We pass the sample to the new leaves, and initialize the _label_average with the value
  data_leaf_node.set_x_t(x_t).set_y_t(y_t);
  other_leaf_node.set_x_t(current_node.x_t()).set_y_t(current_node.y_t());
  // Update downwards of v'
  other_leaf_node.update_downwards(current_node.x_t(), current_node.y_t());
  // Update upwards of v': it's a leaf
  other_leaf_node.update_upwards();
  // Update downwards of v''
  data_leaf_node.update_downwards(x_t, y_t);
  // Note: the update_up of v'' is done in the go_up method, called in fit()
  return data_leaf;
}

template<typename NodeType>
ulong Tree<NodeType>::go_downwards(const ArrayDouble &x_t, double y_t,
                                   bool predict, ulong &depth) {
  // Find the leaf that contains the sample
  // Start at the root. Index of the root is always 0
  // If predict == true, this call to find_leaf is for
  // prediction only, so that no leaf update and splits can be done
  ulong index_current_node = 0;
  depth = 0;
  bool is_leaf = false;
  while (!is_leaf) {
    // Get the current node
    NodeType &current_node = node(index_current_node);
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
      depth++;
    }
  }
  return index_current_node;
}

template<typename NodeType>
void Tree<NodeType>::go_upwards(ulong leaf_index) {
  ulong current = leaf_index;
  while (true) {
    node(current).update_upwards();
    if (current == 0) {
      break;
    }
    // We must update the root node
    current = node(current).parent();
  }
}

template<typename NodeType>
inline ulong Tree<NodeType>::n_nodes() const {
  return _n_nodes;
}

template<typename NodeType>
void Tree<NodeType>::fit(const ArrayDouble &x_t, double y_t) {
  // TODO: Test that the size does not change within successive calls to fit
  if (iteration == 0) {
    nodes[0].set_x_t(x_t).set_y_t(y_t);
    iteration++;
    return;
  }
  ulong depth;
  ulong leaf = go_downwards(x_t, y_t, false, depth);
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

template<typename NodeType>
ulong Tree<NodeType>::add_node(ulong parent, ulong creation_time) {
  nodes.emplace_back(*this, parent);
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
* TreeRegressor methods
*********************************************************************************/

TreeRegressor::TreeRegressor(OnlineForestRegressor &forest)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &tree)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &&tree)
    : Tree<NodeRegressor>(forest) {}

double TreeRegressor::predict(const ArrayDouble &x_t, bool use_aggregation) {
  ulong depth;
  ulong leaf = go_downwards(x_t, 0., true, depth);
  //
  double denominator = -nodes[0].weight_tree();
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
      // weight = current_node.weight() * current_node.predict();
      weight = std::exp(current_node.weight() + denominator / pow(2, depth)) * current_node.predict();
    } else {
//      weight = 0.5 * current_node.weight() * current_node.predict()
//          + 0.5 * node(other).weight_tree() * weight;
      weight = 0.5 * std::exp(current_node.weight() + denominator / pow(2, depth)) * current_node.predict()
          + 0.5 * std::exp(node(other).weight_tree() + denominator / pow(2, depth + 1)) * weight;
    }
    parent = node(current).parent();
    if (node(parent).left() == current) {
      other = node(parent).right();
    } else {
      other = node(parent).left();
    }
    // Root must be updated as well
    if (current == 0) {
      break;
    }
    depth--;
    current = parent;
  }
  // return weight / nodes[0].weight_tree();
  // return weight / std::exp(nodes[0].weight_tree());
  return weight;
}

/*********************************************************************************
* TreeClassifier methods
*********************************************************************************/
TreeClassifier::TreeClassifier(OnlineForestClassifier &forest)
    : Tree<NodeClassifier>(forest) {}

TreeClassifier::TreeClassifier(const TreeClassifier &tree)
    : Tree<NodeClassifier>(forest) {}

TreeClassifier::TreeClassifier(const TreeClassifier &&tree)
    : Tree<NodeClassifier>(forest) {}

double TreeClassifier::predict(const ArrayDouble &x_t, bool use_aggregation) {
  return 0.;
}

inline uint8_t TreeClassifier::n_classes() const {
  return _forest.n_classes();
}

/*********************************************************************************
 * OnlineForest methods
 *********************************************************************************/

template<typename TreeType>
OnlineForest::OnlineForest(uint32_t n_trees,
                           double step,
                           Criterion criterion,
                           int32_t n_threads,
                           int seed,
                           bool verbose)
    : _n_trees(n_trees), _n_threads(n_threads), _criterion(criterion), _step(step), _verbose(verbose), trees() {
  // No iteration so far
  _iteration = 0;
  create_trees();
  // Seed the random number generators
  set_seed(seed);
}

template<typename TreeType>
OnlineForest::~OnlineForest() {}

template<typename TreeType>
void OnlineForest::create_trees() {
  // Just in case...
  trees.clear();
  trees.reserve(_n_trees);
  for (uint32_t i = 0; i < _n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

template<typename TreeType>
void OnlineForest::clear() {
  _iteration = 0;
  create_trees();
}

template<typename TreeType>
void OnlineForest::fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels) {
  ulong n_samples = features->n_rows();
  ulong n_features = features->n_cols();
  set_n_features(n_features);
  for (ulong i = 0; i < n_samples; ++i) {
    for (TreeType &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(view_row(*features, i), (*labels)[i]);
    }
    _iteration++;
  }
}

template<typename TreeType>
inline ulong OnlineForest::sample_feature() {
  return rand.uniform_int(0L, n_features() - 1);
}

template<typename TreeType>
inline double OnlineForest::sample_threshold(double left, double right) {
  return rand.uniform(left, right);
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
    : OnlineForest<TreeRegressor>(n_trees, step, criterion, n_threads, seed, verbose) {}

OnlineForestRegressor::~OnlineForestRegressor() {}

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

//inline double OnlineForestRegressor::step() const {
//  return _step;
//}
//
//void OnlineForestRegressor::print() {
//  for (Tree<NodeRegressor> &tree: trees) {
//    tree.print();
//  }
//}
//
//inline ulong OnlineForestRegressor::n_samples() const {
//  if (_iteration > 0) {
//    return _iteration;
//  } else {
//    TICK_ERROR("You must call ``fit`` before asking for ``n_samples``.")
//  }
//}

//inline ulong OnlineForestRegressor::n_features() const {
//  if (_iteration > 0) {
//    return _n_features;
//  } else {
//    TICK_ERROR("You must call ``fit`` before asking for ``n_features``.")
//  }
//}

//inline OnlineForestRegressor &OnlineForestRegressor::set_n_features(ulong n_features) {
//  if (_iteration == 0) {
//    _n_features = n_features;
//  } else {
//    TICK_ERROR("OnlineForest::set_n_features can be called only once !")
//  }
//  return *this;
//}

//inline uint32_t OnlineForestRegressor::n_trees() const {
//  return _n_trees;
//}


//inline OnlineForestRegressor &OnlineForestRegressor::set_n_trees(uint32_t n_trees) {
//  _n_trees = n_trees;
//  return *this;
//}

//inline int32_t OnlineForestRegressor::n_threads() const {
//  return _n_threads;
//}

//OnlineForestRegressor &OnlineForestRegressor::set_n_threads(int32_t n_threads) {
//  _n_threads = n_threads;
//  return *this;
//}

//inline Criterion OnlineForestRegressor::criterion() const {
//  return _criterion;
//}

//inline OnlineForestRegressor &OnlineForestRegressor::set_criterion(Criterion criterion) {
//  _criterion = criterion;
//  return *this;
//}
//
//inline int OnlineForestRegressor::seed() const {
//  return _seed;
//}

//inline OnlineForestRegressor &OnlineForestRegressor::set_seed(int seed) {
//  _seed = seed;
//  rand.reseed(seed);
//  return *this;
//}

//inline bool OnlineForestRegressor::verbose() const {
//  return _verbose;
//}
//
//inline OnlineForestRegressor &OnlineForestRegressor::set_verbose(bool verbose) {
//  _verbose = verbose;
//  return *this;
//}
