
// License: BSD 3 clause

#include "online_forest.h"

template<typename NodeType>
Node<NodeType>::Node(Tree<NodeType> &tree, ulong index, ulong parent, ulong creation_time)
    : _tree(tree)
//      _samples()
{
  _n_samples = 0;
  _is_leaf = true;
  _left = 0;
  _right = 0;
  _weight = 1;
  _weight_tree = 1;
  // _depth = 0;
  this->_index = index;
  this->_parent = parent;
  this->_time = creation_time;
  // _sample = ArrayDouble(n_features());

//  _features_max = ArrayDouble(n_features());
//  _features_min = ArrayDouble(n_features());
}

template<typename NodeType>
Node<NodeType>::Node(const Node<NodeType> &node)
    : _tree(node._tree), _index(node._index), _left(node._left), _right(node._right),
      _parent(node._parent), _time(node._time),
      _feature(node._feature), _threshold(node._threshold),
      _n_samples(node._n_samples),
      _weight(node._weight), _weight_tree(node._weight_tree),
      _is_leaf(node._is_leaf),
      _sample(node._sample) {
  // std::cout << "Copy const of Node(index: " << _index << ")" << std::endl;
}

template<typename NodeType>
Node<NodeType>::Node(const Node<NodeType> &&node) : _tree(_tree) {
  // std::cout << "Node::Node(const Node && node)" << std::endl;
  _index = node._index;
  _left = node._left;
  _right = node._right;
  _parent = node._parent;
  _time = node._time;
  _feature = node._feature;
  _threshold = node._threshold;
  _n_samples = node._n_samples;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _sample = node._sample;
//  _features_min = node._features_min;
//  _features_max = node._features_max;
//  _samples = node._samples;
}


//template <typename NodeType>
//void Node<NodeType>::update_range(ulong sample_index) {
//  ArrayDouble x_t = get_features(sample_index);
//  // It's the root and the first sample point ever
//  if ((_index == 0) && (_n_samples == 1)) {
//    _features_min = x_t;
//    _features_max = x_t;
//  } else {
//    for (ulong j = 0; j < n_features(); ++j) {
//      double x_tj = x_t[j];
//      if (_features_max[j] < x_tj) {
//        _features_max[j] = x_tj;
//      }
//      if (_features_min[j] > x_tj) {
//        _features_min[j] = x_tj;
//      }
//    }
//  }
//}

template<typename NodeType>
inline ulong Node<NodeType>::index() const {
  return _index;
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
inline const bool Node<NodeType>::is_leaf() const {
  return _is_leaf;
}

template<typename NodeType>
inline Node<NodeType> &Node<NodeType>::set_is_leaf(bool is_leaf) {
  _is_leaf = is_leaf;
  return *this;
}

template<typename NodeType>
inline ulong Node<NodeType>::parent() const {
  return _parent;
}

template<typename NodeType>
inline ulong Node<NodeType>::time() const {
  return _time;
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

//template <typename NodeType>
//inline ulong Node<NodeType>::depth() const {
//  return _depth;
//}
//
//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::set_depth(ulong depth) {
//  _depth = depth;
//  return *this;
//}

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
inline std::pair<ArrayDouble, double> &Node<NodeType>::sample() {
  return _sample;
}


//template <typename NodeType>
//inline const ArrayDouble &Node<NodeType>::features_min() const {
//  return _features_min;
//}
//
//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::set_features_min(const ArrayDouble &features_min) {
//  _features_min = features_min;
//  return *this;
//}
//
//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::set_features_min(const ulong j, const double x) {
//  _features_min[j] = x;
//  return *this;
//}
//
//template <typename NodeType>
//inline const ArrayDouble &Node<NodeType>::features_max() const {
//  return _features_max;
//}
//
//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::set_features_max(const ulong j, const double x) {
//  _features_max[j] = x;
//  return *this;
//}
//
//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::set_features_max(const ArrayDouble &features_max) {
//  _features_max = features_max;
//  return *this;
//}
//
//template <typename NodeType>
//inline const std::vector<ulong> &Node<NodeType>::samples() const {
//  return _samples;
//}
//
//template <typename NodeType>
//inline ulong Node<NodeType>::sample(ulong index) const {
//  return _samples[index];
//}

//template <typename NodeType>
//inline Node<NodeType> &Node<NodeType>::add_sample(ulong index) {
//  _samples.push_back(index);
//  return *this;
//}

template<typename NodeType>
inline Tree<NodeType> &Node<NodeType>::tree() const {
  return _tree;
}

template<typename NodeType>
void Node<NodeType>::print() {
  std::cout << "Node(i: " << _index << ", p: " << _parent
            // << ", f: " << _feature
            // << ", th: " << _threshold
            << ", l: " << _left
            << ", r: " << _right
            // << ", d: " << _depth
            // << ", n: " << n_samples()
            // << ", i: " << _is_leaf
            << ", t: " << _time
            // << ", avg: " << std::setprecision(2) << _labels_average
            // << ", feat_min=[" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
            // << _features_min[1] << "]"
            // << ", feat_max=[" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
            // << _features_max[1] << "]"
            << ")\n";
}

void NodeRegressor::print() {
  std::cout << "Node(idx: " << _index << ", parent: " << _parent
            << ", time: " << _time
            // << ", f: " << _feature
            // << ", th: " << _threshold
            << ", left: " << _left
            << ", right: " << _right
            // << ", d: " << _depth
            // << ", n: " << n_samples()
            // << ", i: " << _is_leaf
            << ", thresh: " << _threshold
            << ", y_hat: " << _labels_average
            << ", sample: ";
  // << ", has_sample:" << _has_sample;
  if (_is_leaf) {
    std::cout << "[" << std::setprecision(2) << _sample.first[0] << ", " << std::setprecision(2) << _sample.first[1]
              << "]";
  } else {
    std::cout << "null";
  }

  std::cout << ", weight: " << _weight;
  std::cout << ", weight_tree: " << _weight_tree;
  std::cout << ")\n";
  // << ", avg: " << std::setprecision(2) << _labels_average
  // << ", feat_min=[" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
  // << _features_min[1] << "]"
  // << ", feat_max=[" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
  // << _features_max[1] << "]"
}

template<typename NodeType>
Node<NodeType>::~Node() {}

NodeRegressor::NodeRegressor(Tree<NodeRegressor> &tree, ulong index, ulong parent, ulong creation_time)
    : Node(tree, index, parent, creation_time) {
  _labels_average = 0;
  // std::cout << "Creating node regressor" << std::endl;
}

NodeRegressor::NodeRegressor(const NodeRegressor &node)
    : Node(node), _labels_average(node._labels_average) {}

NodeRegressor::NodeRegressor(const NodeRegressor &&node)
    : Node(node) {
  _labels_average = node._labels_average;
}

NodeRegressor::~NodeRegressor() {}

inline double NodeRegressor::labels_average() const {
  return _labels_average;
}

inline NodeRegressor &NodeRegressor::set_labels_average(double avg) {
  _labels_average = avg;
  return *this;
}

template<typename NodeType>
void Node<NodeType>::update_down(const ArrayDouble &x_t, double y_t) {
  _n_samples++;
  // TODO: Make compute loss virtual insteal
  update_weight(x_t, y_t);
  update_label_stats(y_t);
}

void NodeRegressor::update_label_stats(double y_t) {
  // When a node is updated, it necessarily contains already a sample
  _labels_average = ((_n_samples - 1) * _labels_average + y_t) / _n_samples;
}

void NodeRegressor::update_weight(const ArrayDouble &x_t, double y_t) {
  double diff = _labels_average - y_t;
  _weight *= exp(-diff * diff / 2);
  // _weight -= diff * diff / 2;
}

NodeRegressor &NodeRegressor::set_sample(const ArrayDouble &x_t, double y_t) {
  _sample = std::pair<ArrayDouble, double>(x_t, y_t);
  // TODO: Don't do this line !!!
  // _labels_average = y_t;
  return *this;
}

NodeRegressor &NodeRegressor::set_sample(const std::pair<ArrayDouble, double> &sample) {
  return set_sample(sample.first, sample.second);
}

template<typename NodeType>
std::pair<ulong, double> Tree<NodeType>::sample_feature_and_threshold(ulong index) {
  if (criterion() == Criterion::unif) {
    // Select the splitting feature uniformly at random
    ulong feature = forest.sample_feature();
    // Choose at random the feature used to cut uniformly in the
    // range of the features
    double left_boundary = node(index).features_min()[feature];
    double right_boundary = node(index).features_max()[feature];
    double threshold = forest.sample_threshold(left_boundary, right_boundary);
    return std::pair<ulong, double>(feature, threshold);
  } else {
    TICK_ERROR("Criterion::mse not implemented.");
  }
};

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &tree)
    : nodes(tree.nodes), forest(tree.forest), already_fitted(tree.already_fitted) {
  // std::cout << "Tree::Tree(const &Tree tree)" << std::endl;
}

template<typename NodeType>
Tree<NodeType>::Tree(const Tree<NodeType> &&tree) : nodes(tree.nodes), forest(tree.forest) {
  already_fitted = tree.already_fitted;
}

template<typename NodeType>
ulong Node<NodeType>::n_features() const {
  return _tree.n_features();
}

//template <typename NodeType>
//double Node<NodeType>::get_label(ulong sample_index) const {
//  return _tree.get_label(sample_index);
//}
//
//template <typename NodeType>
//ArrayDouble Node<NodeType>::get_features(ulong sample_index) const {
//  return _tree.get_features(sample_index);
//}

template<typename NodeType>
Tree<NodeType>::Tree(OnlineForestRegressor &forest) : forest(forest) {
  // std::cout << "Tree::Tree(OnlineForest &forest)\n";
  // std::cout << "Creating the root" << std::endl;
  add_node(0, 0);
  // TODO: pre-allocate the vector to make things faster ?
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
ulong Tree<NodeType>::go_down(const ArrayDouble &x_t, double y_t, bool predict) {
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
      current_node.update_down(x_t, y_t);
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
void Tree<NodeType>::go_up(ulong leaf_index) {
  // TODO: IL FAUT BIEN UPDATED AUSSI LA RACINE ???
  ulong current = leaf_index;
  while (true) {
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
    current = node(current).parent();
    if (current_node.index() == 0) {
      break;
    }
  }
}

template<typename NodeType>
void Tree<NodeType>::rescale() {
  double scale = 1 / nodes[0].weight();
  for(NodeType &node : nodes) {
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


  ulong leaf = go_down(x_t, y_t, false);
  // ulong new_leaf = leaf;

  ulong new_leaf = split_leaf(leaf, x_t, y_t);

//  for(ulong j=0; j < n_features(); ++j) {
//    double delta = std::abs(x_t[j] - node(leaf).sample().first[j]);
//    if (delta > 0.) {
//      new_leaf = split_node(leaf, x_t, y_t);
//      break;
//    }
//  }
  // TODO: question for Jaouad, if we don't split the node, do we need to go backwards ?

  go_up(new_leaf);

  //  std::cout << "iteration: " << iteration << ", x_t: [";
  //  std::cout << std::setprecision(2) << x_t[0] << ", ";
  //  std::cout << std::setprecision(2) << x_t[1] << "]";
  //  std::cout << std::setprecision(2) << ", y_t: " << y_t << std::endl;
  //  std::cout << "leaf_index= " << leaf_index << std::endl;
  //  if (node(leaf_index).is_leaf()) {
  // std::cout << "n_features()= " << n_features() << std::endl;
  // std::cout << "max= " << max << std::endl;
  // std::cout << "node(leaf_index).set_sample(x_t);" << std::endl;
  // node(leaf_index).set_sample(x_t);
  // }
  // print();
  iteration++;
}

TreeRegressor::TreeRegressor(OnlineForestRegressor &forest)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &tree)
    : Tree<NodeRegressor>(forest) {}

TreeRegressor::TreeRegressor(const TreeRegressor &&tree)
    : Tree<NodeRegressor>(forest) {}

double TreeRegressor::predict(const ArrayDouble &x_t, bool use_aggregation) {
  ulong leaf = go_down(x_t, 0., true);
  if(!use_aggregation) {
    return node(leaf).sample().second;
  }
  ulong current = leaf;
  // The child of the current node that does not contain the data
  ulong other;
  ulong parent;
  double weight;
  while (true) {
    NodeRegressor &current_node = node(current);
    if (current_node.is_leaf()) {
      weight = current_node.weight() * current_node.labels_average();
      // weight = std::exp(current_node.weight()) * current_node.labels_average();
    } else {
      weight = 0.5 * current_node.weight() * current_node.labels_average()
          + 0.5 * node(other).weight_tree() *  weight;
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
    if (current_node.index() == 0) {
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

OnlineForestRegressor::OnlineForestRegressor(uint32_t n_trees,
                                             Criterion criterion,
    // int32_t max_depth,
    // uint32_t min_samples_split,
                                             int32_t n_threads,
                                             int seed,
                                             bool verbose)
    : trees(), _n_trees(n_trees), _criterion(criterion),
    // _max_depth(max_depth),
    // _min_samples_split(min_samples_split),
      _n_threads(n_threads),
      _verbose(verbose) {
  // _n_splits(n_splits) {
  // _fitted = false;

  // No iteration so far
  _iteration = 0;
  _n_features_known = false;

  // std::cout << "create_trees();" << std::endl;
  create_trees();

  // permutation_ready = false;
  // rand = Rand(123);
  // cycle_type = CycleType::sequential;
  // i_perm = 0;

  // Seed the random number generators
  set_seed(seed);
}

// Do n_iter iterations


void OnlineForestRegressor::create_trees() {
  // Just in case...
  trees.clear();
  trees.reserve(_n_trees);
  for (uint32_t i = 0; i < _n_trees; ++i) {
    // std::cout << "add tree" << i << std::endl;
    trees.emplace_back(*this);
  }
}

void OnlineForestRegressor::fit(const SArrayDouble2dPtr features,
                                const SArrayDoublePtr labels) {
  ulong n_samples = features->n_rows();
  ulong n_features = features->n_cols();
  set_n_features(n_features);
  for (ulong i = 0; i < n_samples; ++i) {
    // ulong sample_index = get_next_sample();
    for (TreeRegressor &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(view_row(*features, i), (*labels)[i]);
    }
    _iteration++;
  }
}

//void OnlineForestRegressor::set_data(const SArrayDouble2dPtr features,
//                                     const SArrayDoublePtr labels) {
//  this->_features_fit = features;
//  this->_labels_fit = labels;
//  _fitted = true;
//  trees.clear();
//  // TODO: when we set_data, we need to recreate the trees
//  for (uint32_t i = 0; i < _n_trees; ++i) {
//    trees.emplace_back(*this);
//  }
//}

void OnlineForestRegressor::predict(const SArrayDouble2dPtr features,
                                    SArrayDoublePtr predictions, bool use_aggregation) {
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
    TICK_ERROR("OnlineRandomForest::predict: you must fit first.")
  }
}

//void OnlineForestRegressor::init_permutation() {
//  if ((cycle_type == CycleType::permutation) && (n_samples() > 0)) {
//    permutation = ArrayULong(n_samples());
//    for (ulong i = 0; i < n_samples(); ++i)
//      permutation[i] = i;
//  }
//}
//
////// Simulation of a random permutation using Knuth's algorithm
//void OnlineForestRegressor::shuffle() {
//  if (cycle_type == CycleType::permutation) {
//    // A secure check
//    if (permutation.size() != n_samples()) {
//      init_permutation();
//    }
//    // Restart the i_perm
//    i_perm = 0;
//    for (ulong i = 1; i < n_samples(); ++i) {
//      // uniform number in { 0, ..., i }
//      ulong j = rand.uniform_int(0L, i);
//      // Exchange permutation[i] and permutation[j]
//      ulong tmp = permutation[i];
//      permutation[i] = permutation[j];
//      permutation[j] = tmp;
//    }
//  }
//  permutation_ready = true;
//}
//
//
//ulong OnlineForestRegressor::get_next_sample() {
//  ulong i = 0;
//  if (cycle_type == CycleType::uniform) {
//    i = rand.uniform_int(0L, n_samples() - 1);
//  } else {
//    if (cycle_type == CycleType::permutation) {
//      if (!permutation_ready) {
//        shuffle();
//      }
//      i = permutation[i_perm];
//      i_perm++;
//      if (i_perm >= n_samples()) {
//        shuffle();
//      }
//    } else {
//      // Otherwise it's cycling through the data
//      i = i_perm;
//      i_perm++;
//      if (i_perm >= n_samples()) {
//        i_perm = 0;
//      }
//    }
//  }
//  return i;
//}


//
//template <typename NodeType>
//inline ArrayDouble Tree<NodeType>::get_features(ulong sample_index) const {
//  return forest.features(sample_index);
//}
//
//template <typename NodeType>
//inline ArrayDouble Tree<NodeType>::get_features_predict(ulong sample_index) const {
//  return forest.features_predict(sample_index);
//}
//
//template <typename NodeType>
//inline uint32_t Tree<NodeType>::min_samples_split() const {
//  return forest.min_samples_split();
//}

template<typename NodeType>
inline ulong Tree<NodeType>::n_features() const {
  return forest.n_features();
}

//template <typename NodeType>
//inline double Tree<NodeType>::get_label(ulong sample_index) const {
//  return forest.label(sample_index);
//}

template<typename NodeType>
inline Criterion Tree<NodeType>::criterion() const {
  return forest.criterion();
}
