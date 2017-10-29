
// License: BSD 3 clause

#include "online_forest.h"

Node::Node(Tree &tree, ulong index, ulong parent, ulong creation_time)
    : in_tree(tree), _samples() {
  _n_samples = 0;
  _is_leaf = true;
  _left = 0;
  _right = 0;
  this->_index = index;
  this->_parent = parent;
  this->_creation_time = creation_time;
  _features_max = ArrayDouble(n_features());
  _features_min = ArrayDouble(n_features());
}

Node::Node(const Node &node)
    : in_tree(node.in_tree), _index(node._index), _left(node._left), _right(node._right),
      _parent(node._parent), _creation_time(node._creation_time),
      _feature(node._feature), _threshold(node._threshold),
      _impurity(node._impurity), _n_samples(node._n_samples), _labels_average(node._labels_average),
      _aggregation_weight(node._aggregation_weight), _aggregation_weight_ctw(node._aggregation_weight_ctw),
      _features_min(node._features_min), _features_max(node._features_max),
      _samples(node._samples), _is_leaf(node._is_leaf) {
  // std::cout << "Copy const of Node(index: " << _index << ")" << std::endl;
}

Node::Node(const Node &&node) : in_tree(in_tree) {
  // std::cout << "Node::Node(const Node && node)" << std::endl;
  _index = node._index;
  _left = node._left;
  _right = node._right;
  _parent = node._parent;
  _creation_time = node._creation_time;
  _feature = node._feature;
  _threshold = node._threshold;
  _impurity = node._impurity;
  _n_samples = node._n_samples;
  _labels_average = node._labels_average;
  _aggregation_weight = node._aggregation_weight;
  _aggregation_weight_ctw = node._aggregation_weight_ctw;
  _is_leaf = node._is_leaf;
  _features_min = node._features_min;
  _features_max = node._features_max;
  _samples = node._samples;
}


void Node::update(ulong sample_index, bool update_range) {
  _n_samples++;
  ArrayDouble x_t = get_features(sample_index);
  double y_t = get_label(sample_index);
  if (_is_leaf) {
    // It's the root and the first sample point ever
    if ((_index == 0) && (_n_samples == 1)) {
      if (update_range) {
        _features_min = x_t;
        _features_max = x_t;
      }
      _labels_average = y_t;
    } else {
      if (update_range) {
        for(ulong j=0; j < n_features(); ++j) {
          double x_tj = x_t[j];
          if (_features_max[j] < x_tj) {
            _features_max[j] = x_tj;
          }
          if (_features_min[j] > x_tj) {
            _features_min[j] = x_tj;
          }
        }
      }
      // Update the average of labels online
      _labels_average = ((_n_samples - 1) * _labels_average + y_t) / _n_samples;
    }
    // If it's a leaf, save the sample
    _samples.push_back(sample_index);
  }
}


void Tree::split_node(ulong index) {
  // Choose at random the feature used to cut
  ulong left = add_node(index, iteration);
  ulong right = add_node(index, iteration);
  // Give back information about the childs to the parent node
  nodes[index].set_left(left).set_right(right).set_is_leaf(false);

  // Select the splitting feature uniformly at random
  ulong feature = sample_feature_uniform();
  // Select the thresholding uniformly at random in the support of the current features
  double left_boundary = nodes[index].features_min()[feature];
  double right_boundary = nodes[index].features_max()[feature];
  double threshold = sample_threshold_uniform(left_boundary, right_boundary);
  nodes[index].set_feature(feature).set_threshold(threshold);

  // The ranges of the childs is contained in the range of the parent
  // We first simply copy the ranges, and update the feature using the selected
  // threshold
  nodes[left].set_features_min(nodes[index].features_min());
  nodes[left].set_features_max(nodes[index].features_max());
  nodes[left].set_features_max(feature, threshold);
  nodes[right].set_features_min(nodes[index].features_min());
  nodes[right].set_features_max(nodes[index].features_max());
  nodes[right].set_features_min(feature, threshold);

  // Split the samples of the parent and update the childs using them
  // This is maybe some kind of cheating...
  for (ulong i = 0; i < nodes[index].n_samples(); ++i) {
    ulong sample_index = nodes[index].sample(i);
    double x_ij = get_features(sample_index)[feature];
    if(x_ij <= threshold) {
      // We don't update the ranges, since we already managed them above
      nodes[left].update(sample_index, false);
    } else {
      nodes[right].update(sample_index, false);
    }
  }
}

const Node &Tree::get_node(ulong node_index) const {
  return nodes[node_index];
}

Tree::Tree(const Tree &tree)
    : nodes(tree.nodes), forest(tree.forest), already_fitted(tree.already_fitted) {
  // std::cout << "Tree::Tree(const &Tree tree)" << std::endl;
}

Tree::Tree(const Tree &&tree) : forest(tree.forest), nodes(tree.nodes) {
  already_fitted = tree.already_fitted;
}

OnlineForest &Tree::get_forest() const {
  return forest;
}

double Node::get_label(ulong sample_index) const {
  return in_tree.get_label(sample_index);
}

ulong Node::n_features() const {
  return in_tree.n_features();
}

ArrayDouble Node::get_features(ulong sample_index) const {
  return in_tree.get_features(sample_index);
}

Tree::Tree(OnlineForest &forest) : forest(forest) {
  // std::cout << "Tree::Tree(OnlineForest &forest)\n";
  add_node(0, 0);
}


ulong Tree::find_leaf(ulong sample_index, bool predict) {
  // Find the leaf that contains the sample
  // Start at the root. Index of the root is always 0
  ulong index_current_node = 0;
  bool is_leaf = false;
  ArrayDouble x_t;
  if (predict) {
    x_t = get_features_predict(sample_index);
  } else {
    x_t = get_features(sample_index);
  }
  while (!is_leaf) {
    // Get the current node
    Node &current_node = nodes[index_current_node];
    if (predict) {
      current_node.update(sample_index);
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

void Tree::fit(ulong sample_index) {
  // TODO: Test that the size does not change within successive calls to fit
  iteration++;
  ulong leaf_index = find_leaf(sample_index, true);
  if (nodes[leaf_index].n_samples() >= n_min_samples()) {
    split_node(leaf_index);
  }
  // print();
}

double Tree::predict(ulong sample_index) {
  ulong leaf_index = find_leaf(sample_index, false);
  return nodes[leaf_index].labels_average();
}

ulong Tree::add_node(ulong parent, ulong creation_time) {
  nodes.emplace_back(*this, n_nodes, parent, creation_time);
  return n_nodes++;
}

OnlineForest::OnlineForest(uint32_t n_trees, uint32_t n_min_samples,
                           uint32_t n_splits)
    : n_trees(n_trees), _n_min_samples(n_min_samples),
      _n_splits(n_splits), trees(), rand(), rand_feature(), rand_threshold() {
  has_data = false;
  // No iteration so far
  t = 0;
  //
  permutation_ready = false;
  // rand = Rand(123);
  cycle_type = CycleType::sequential;
  i_perm = 0;
  trees.reserve(n_trees);
}

// Do n_iter iterations
void OnlineForest::fit(ulong n_iter) {
  if (!has_data) {
    TICK_ERROR("OnlineForest::fit: the forest has no data yet.")
  }
  // Could be parallelized
  if (n_iter == 0) {
    n_iter = n_samples();
  }
  for (ulong it = 0; it < n_iter; ++it) {
    if (it % n_samples() == 0) {
      // std::cout << "pass number: " << it / n_samples() << std::endl;
    }
    ulong sample_index = get_next_sample();
    for (Tree &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(sample_index);
    }
    t++;
  }
}

void OnlineForest::init_permutation() {
  if ((cycle_type == CycleType::permutation) && (n_samples() > 0)) {
    permutation = ArrayULong(n_samples());
    for (ulong i = 0; i < n_samples(); ++i)
      permutation[i] = i;
  }
}

//// Simulation of a random permutation using Knuth's algorithm
void OnlineForest::shuffle() {
  if (cycle_type == CycleType::permutation) {
    // A secure check
    if (permutation.size() != n_samples()) {
      init_permutation();
    }
    // Restart the i_perm
    i_perm = 0;
    for (ulong i = 1; i < n_samples(); ++i) {
      // uniform number in { 0, ..., i }
      ulong j = rand_unif(i);
      // Exchange permutation[i] and permutation[j]
      ulong tmp = permutation[i];
      permutation[i] = permutation[j];
      permutation[j] = tmp;
    }
  }
  permutation_ready = true;
}

ulong OnlineForest::get_next_sample() {
  ulong i = 0;
  if (cycle_type == CycleType::uniform) {
    i = rand_unif(n_samples() - 1);
  } else {
    if (cycle_type == CycleType::permutation) {
      if (!permutation_ready) {
        shuffle();
      }
      i = permutation[i_perm];
      i_perm++;
      if (i_perm >= n_samples()) {
        shuffle();
      }
    } else {
      // Otherwise it's cycling through the data
      i = i_perm;
      i_perm++;
      if (i_perm >= n_samples()) {
        i_perm = 0;
      }
    }
  }
  return i;
}

void OnlineForest::set_data(const SArrayDouble2dPtr features,
                            const SArrayDoublePtr labels) {
  this->_features_fit = features;
  this->_labels_fit = labels;
  has_data = true;
  trees.clear();
  // TODO: when we set_data, we need to recreate the trees
  for (uint32_t i = 0; i < n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

void OnlineForest::predict(const SArrayDouble2dPtr features,
                           SArrayDoublePtr predictions) {
  // TODO: check that the forest is already trained
  this->_features_predict = features;
  ulong n_samples = features->n_rows();
  for (ulong i=0; i < n_samples; ++i) {
    // The prediction is simply the average of the predictions
    double y_pred = 0;
    for (Tree &tree : trees) {
      y_pred += tree.predict(i);
    }
    (*predictions)[i] = y_pred / n_samples;
  }
}

inline ArrayDouble Tree::get_features(ulong sample_index) const {
  return forest.features(sample_index);
}

inline ArrayDouble Tree::get_features_predict(ulong sample_index) const {
  return forest.features(sample_index);
}

inline uint32_t Tree::n_min_samples() const {
  return forest.n_min_samples();
}

inline ulong Tree::n_features() const {
  return forest.n_features();
}

inline double Tree::get_label(ulong sample_index) const {
  return forest.label(sample_index);
}

inline ulong Tree::sample_feature_uniform() {
  return forest.sample_feature_uniform();
}

inline double Tree::sample_threshold_uniform(double left, double right) {
  return forest.sample_threshold_uniform(left, right);
}
