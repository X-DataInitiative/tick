// License: BSD 3 clause

#include "tick/online/online_forest_classifier.h"

/*********************************************************************************
 * NodeClassifier methods
 *********************************************************************************/

// LDFLAGS="-L/anaconda3/envs/py36/lib" ./sh/mkn.sh online

NodeClassifier::NodeClassifier(TreeClassifier &tree, uint32_t index, uint32_t parent, float time)
    : _tree(tree),
      _index(index),
      _parent(parent),
      _time(time),
      _counts(tree.n_classes()) {
  _counts.fill(0);
  _samples = std::vector<uint32_t>();
  _memory_range_min = std::vector<float>();
  _memory_range_max = std::vector<float>();
}


NodeClassifier::NodeClassifier(const NodeClassifier &node)
    : _tree(node._tree),
      _index(node._index),
      _is_leaf(node._is_leaf),
      _depth(node._depth),
      _n_samples(node._n_samples),
      _parent(node._parent),
      _left(node._left),
      _right(node._right),
      _feature(node._feature),
      _y_t(node._y_t),
      _weight(node._weight),
      _weight_tree(node._weight_tree),
      _threshold(node._threshold),
      _time(node._time),
      _counts(node._counts),
      _memory_range_min(node._memory_range_min),
      _memory_range_max(node._memory_range_max),
      _samples(node._samples),
      _memorized(node._memorized) {
}


NodeClassifier::NodeClassifier(const NodeClassifier &&node)
    : _tree(node._tree) {
  _index = node._index;
  _parent = node._parent;
  _left = node._left;
  _right = node._right;
  _feature = node._feature;
  _threshold = node._threshold;
  _time = node._time;
  _depth = node._depth;
  _memory_range_min = node._memory_range_min;
  _memory_range_max = node._memory_range_max;
  _samples = node._samples;
  _n_samples = node._n_samples;
  _y_t = node._y_t;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _counts = node._counts;
  _memorized = node._memorized;
}


NodeClassifier &NodeClassifier::operator=(const NodeClassifier &node) {
  // We absolutely shall not copy _index !
  _parent = node._parent;
  _left = node._left;
  _right = node._right;
  _feature = node._feature;
  _threshold = node._threshold;
  _time = node._time;
  _depth = node._depth;
  _memory_range_min = node._memory_range_min;
  _memory_range_max = node._memory_range_max;
  _samples = node._samples;
  _n_samples = node._n_samples;
  _y_t = node._y_t;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _counts = node._counts;

  // If the tree isn't full, this is not memorized and node is, then we this to memorize this
  if (!_memorized && node._memorized && !_tree.is_full()) {
    memorize_range();
  } else {
    // Warning: range has not been memorized !
    _memorized = false;
  }
  return *this;
}


float NodeClassifier::update_weight(const ArrayFloat &x_t, const float y_t) {
  float loss_t = loss(y_t);
  if (use_aggregation()) {
    _weight -= step() * loss_t;
  }
  // We return the loss before updating the predictor of the node in order to
  // update the feature importance in TreeClassifier::go_downwards
  return loss_t;
}

void NodeClassifier::update_count(const float y_t) {
  // We update the counts for the class y_t
  _counts[static_cast<uint8_t>(y_t)]++;
}

// TODO: actually do_update_weight is always false if the node is a leaf, otherwise it's true
void NodeClassifier::update_downwards(uint32_t sample, bool do_update_weight) {
  // Get the sample features and label
  const ArrayFloat& x_t = sample_features(sample);
  float y_t = sample_label(sample);
  if (_memorized) {
    // If range is memorized, we update it
    for (ulong j = 0; j < n_features(); ++j) {
      float x_tj = x_t[j];
      if (x_tj < _memory_range_min[j]) {
        _memory_range_min[j] = x_tj;
      }
      if (x_tj > _memory_range_max[j]) {
        _memory_range_max[j] = x_tj;
      }
    }
  }
  // Now, we actually add the sample
  _samples.emplace_back(sample);
  // We node range is memorized, then we need to inform as well the tree
  if (_memorized) {
    _tree.incr_n_samples(_n_samples, _index);
  }
  _n_samples++;
  if (do_update_weight) {
    update_weight(x_t, y_t);
  }
  update_count(y_t);
  // Now that the node is updated, we update also its range type
  _tree.update_range_type(_n_samples, _index);
}


void NodeClassifier::update_weight_tree() {
  if (_is_leaf) {
    _weight_tree = _weight;
  } else {
    _weight_tree = log_sum_2_exp(_weight, node(_left).weight_tree() + node(_right).weight_tree());
  }
}

bool NodeClassifier::is_dirac(const float y_t) {
  // Returns true if the node only has the same label as y_t
  return (_n_samples == _counts[static_cast<uint8_t>(y_t)]);
}

uint32_t NodeClassifier::get_child(const ArrayFloat &x_t) {
  if (x_t[_feature] <= _threshold) {
    return _left;
  } else {
    return _right;
  }
}

float NodeClassifier::score(uint8_t c) const {
  // Using the Dirichet prior
  return (_counts[c] + dirichlet()) / (_n_samples + dirichlet() * n_classes());
}

void NodeClassifier::compute_range_extension(const ArrayFloat &x_t, ArrayFloat &extensions,
                                             float &extensions_sum, float &extensions_max) {
  extensions_sum = 0;
  extensions_max = std::numeric_limits<float>::lowest();
  for (uint32_t j = 0; j < n_features(); ++j) {
    float x_tj = x_t[j];
    float feature_min_j, feature_max_j;
    std::tie(feature_min_j, feature_max_j) = range(j);
    float diff;
    if (x_tj < feature_min_j) {
      diff = feature_min_j - x_tj;
    } else {
      if (x_tj > feature_max_j) {
        diff = x_tj - feature_max_j;
      } else {
        diff = 0;
      }
    }
    extensions[j] = diff;
    extensions_sum += diff;
    if (diff > extensions_max) {
      extensions_max = diff;
    }
  }
}

void NodeClassifier::predict(ArrayFloat &scores) const {
  for (uint8_t c = 0; c < n_classes(); ++c) {
    scores[c] = score(c);
  }
}

float NodeClassifier::loss(const float y_t) {
  // Log-loss
  uint8_t c = static_cast<uint8_t>(y_t);
  return -std::log(score(c));
}

inline NodeClassifier &NodeClassifier::node(uint32_t index) const { return _tree.node(index); }

uint32_t NodeClassifier::n_features() const { return _tree.n_features(); }

uint8_t NodeClassifier::n_classes() const { return _tree.n_classes(); }

inline float NodeClassifier::step() const { return _tree.step(); }

inline float NodeClassifier::dirichlet() const { return _tree.dirichlet(); }

inline uint32_t NodeClassifier::parent() const { return _parent; }
inline NodeClassifier &NodeClassifier::parent(uint32_t parent) {
  _parent = parent;
  return *this;
}

inline uint32_t NodeClassifier::left() const { return _left; }
inline NodeClassifier &NodeClassifier::left(uint32_t left) {
  _left = left;
  return *this;
}

inline uint32_t NodeClassifier::right() const { return _right; }
inline NodeClassifier &NodeClassifier::right(uint32_t right) {
  _right = right;
  return *this;
}

inline bool NodeClassifier::is_leaf() const { return _is_leaf; }
inline NodeClassifier &NodeClassifier::is_leaf(bool is_leaf) {
  _is_leaf = is_leaf;
  return *this;
}

inline uint32_t NodeClassifier::feature() const { return _feature; }
inline NodeClassifier &NodeClassifier::feature(uint32_t feature) {
  _feature = feature;
  return *this;
}

inline float NodeClassifier::threshold() const { return _threshold; }
inline NodeClassifier &NodeClassifier::threshold(float threshold) {
  _threshold = threshold;
  return *this;
}

inline float NodeClassifier::time() const { return _time; }
inline NodeClassifier &NodeClassifier::time(float time) {
  _time = time;
  return *this;
}

inline uint8_t NodeClassifier::depth() const { return _depth; }
inline NodeClassifier &NodeClassifier::depth(uint8_t depth) {
  _depth = depth;
  return *this;
}


inline bool NodeClassifier::use_aggregation() const { return _tree.use_aggregation(); }

inline float NodeClassifier::weight() const { return _weight; }

inline float NodeClassifier::weight_tree() const { return _weight_tree; }

inline const ArrayFloat& NodeClassifier::sample_features(uint32_t sample) const {
  return _tree.sample_features(sample);
}

inline float NodeClassifier::sample_label(uint32_t sample) const {
  return _tree.sample_label(sample);
}

void NodeClassifier::print() {
  std::cout << "Node(index: " << _index << ", parent: " << _parent << ", left: " << _left << ", right: " << _right
            << ", time: " << std::setprecision(2) << _time << ", n_samples: " << _n_samples
            << ", is_leaf: " << _is_leaf;
            // << ", scores: [" << std::setprecision(2) << score(0)
            // << ", " << std::setprecision(2) << score(1) << "]"
            // << ", counts: [" << std::setprecision(2) << _counts[0] << ", " << std::setprecision(2)
            // << _counts[1] << "], "
            // << ", fmin: [" << std::setprecision(2) << features_min(0) << ", "
            // << std::setprecision(2) << range(features_min(1) << "], "
            // << ", fmax: [" << std::setprecision(2) << features_max(0) << ", "
            // << std::setprecision(2) << features_max(1) << "]";
  std::cout << ")\n";
}


std::pair<float, float> NodeClassifier::range(uint32_t j) const {
  if (_n_samples == 0) {
    // TODO: weird to have to do this
    TICK_ERROR("Node has no range since it has no samples")
  } else {
    if (_memorized) {
      // If range is memorized, then we return it directly
      return std::pair<float, float>(_memory_range_min[j], _memory_range_max[j]);
    } else {
      // Otherwise, we need to compute it from the samples features
      if (_n_samples == 1) {
        uint32_t sample = _samples.front();
        float feature_j = sample_features(sample)[j];
        return std::pair<float, float>(feature_j, feature_j);
      } else {
        float x_0j = sample_features(_samples.front())[j];
        float range_min = x_0j;
        float range_max = x_0j;
        for(auto sample_iter = std::next(_samples.begin()); sample_iter != _samples.end(); ++sample_iter) {
          float x_tj = sample_features(*sample_iter)[j];
          if (x_tj < range_min) {
            range_min = x_tj;
          }
          if (x_tj > range_max) {
            range_max = x_tj;
          }
        }
        return std::pair<float, float>(range_min, range_max);
      }
    }
  }
}


void NodeClassifier::forget_range() {
  if (_memorized) {
    _memory_range_min.clear();
    _memory_range_max.clear();
    // Don't forget to inform the tree about this node
    _tree.remove_from_disposables(_n_samples, _index);
    _memorized = false;
  }
}

void NodeClassifier::memorize_range() {
  if(!_memorized && _n_samples >= 2 && !_tree.is_full()) {
    // We memorize range if it is not memorized yet, if it has more than one sample and if the
    // tree is not full
    if (_memory_range_min.empty()) {
      // TODO: new one or reserve ?
      // _memory_range_min.reserve(n_features());
      _memory_range_min = std::vector<float>(n_features());
      _memory_range_max = std::vector<float>(n_features());
    }
    if(!_samples.empty()) {
      // First, copy the first sample into the range
      float *begin = sample_features(_samples.front()).data();
      float *end = begin + n_features();
      std::copy(begin, end, _memory_range_min.begin());
      std::copy(begin, end, _memory_range_max.begin());
      // Then, update the range using the rest of the points
      for(auto sample_iter= std::next(_samples.begin()); sample_iter != _samples.end(); ++sample_iter) {
        const ArrayFloat & x_t = sample_features(*sample_iter);
        for (uint32_t j=0; j < n_features(); ++j) {
          float x_tj = x_t[j];
          if (x_tj < _memory_range_min[j]) {
            _memory_range_min[j] = x_tj;
          }
          if (x_tj > _memory_range_max[j]) {
            _memory_range_max[j] = x_tj;
          }
        }
      }
    }
    _memorized = true;
    // Don't forget to inform the tree about this node
    _tree.add_to_disposables(_n_samples, _index);
  }
}



/*********************************************************************************
 * TreeClassifier methods
 *********************************************************************************/

TreeClassifier::TreeClassifier(OnlineForestClassifier &forest)
    : forest(forest), _n_features(forest.n_features()), _n_classes(forest.n_classes()) {
  create_root();
  feature_importances_ = ArrayFloat(_n_features);
  intensities = ArrayFloat(_n_features);
  // TODO: initialization might be important
  feature_importances_.fill(1.);

}


TreeClassifier::TreeClassifier(const TreeClassifier &tree)
    : forest(tree.forest), nodes(tree.nodes) {}

TreeClassifier::TreeClassifier(const TreeClassifier &&tree)
    : forest(tree.forest), nodes(tree.nodes) {}


void TreeClassifier::fit(uint32_t sample) {
  uint32_t leaf = go_downwards(sample);
  if (use_aggregation()) {
    go_upwards(leaf);
  }
  iteration++;
}


uint32_t TreeClassifier::go_downwards(uint32_t sample) {
  // We update the nodes along the path which leads to the leaf containing x_t
  // For each node on the path, we consider the possibility of splitting it,
  // following the Mondrian process definition.
  // Index of the root is 0
  uint32_t index_current_node = 0;
  bool is_leaf = false;
  const ArrayFloat& x_t = sample_features(sample);
  if (iteration == 0) {
    // If it's the first iteration, we just put the point in the range of root
    // Let's get the root
    NodeClassifier &current_node = node(0);
    current_node.update_downwards(sample, false);
    return index_current_node;
  } else {
    while (true) {
      // If it's not the first iteration (otherwise the current node is root
      // with no range), we consider the possibility of a split
      float split_time = compute_split_time(index_current_node, sample);
      if (split_time > 0) {
        // We split the current node: because the current node is a leaf, or
        // because we add a new node along the path
        NodeClassifier &current_node = node(index_current_node);
        // Normalize the range extensions to get probabilities
        intensities /= intensities.sum();
        // Sample the feature at random with with a probability proportional to
        // the range extensions
        uint32_t feature = forest.sample_feature(intensities);
        // Value of the feature
        float x_tf = x_t[feature];
        // Is it a right extension of the node ?
        float range_min, range_max;
        std::tie(range_min, range_max) = current_node.range(feature);
        bool is_right_extension = x_tf > range_max;
        float threshold;
        if (is_right_extension) {
          threshold = forest.sample_threshold(range_max, x_tf);
        } else {
          threshold = forest.sample_threshold(x_tf, range_min);
        }
        split_node(index_current_node, split_time, threshold, feature, is_right_extension);
        // Update the depth of the new childs of the current node
        // Get the current node again, and update it
        NodeClassifier &current_node_again = node(index_current_node);

        current_node_again.update_downwards(sample, true);

        uint32_t left = current_node_again.left();
        uint32_t right = current_node_again.right();
        uint8_t depth = current_node_again.depth();

        // Now, get the next node
        if (is_right_extension) {
          index_current_node = right;
        } else {
          index_current_node = left;
        }
        // TODO: we can use get_child ?

        update_depth(left, depth);
        update_depth(right, depth);

        // This is the leaf containing the sample point (we've just splitted the
        // current node with the data point)
        NodeClassifier &leaf = node(index_current_node);
        // Let's update the leaf containing the point
        leaf.update_downwards(sample, false);
        return index_current_node;
      } else {
        // There is no split, so we just update the node and go to the next one
        NodeClassifier &current_node = node(index_current_node);
        current_node.update_downwards(sample, true);
        is_leaf = current_node.is_leaf();
        if (is_leaf) {
          return index_current_node;
        } else {
          index_current_node = current_node.get_child(x_t);
        }
      }
    }
  }
}


void TreeClassifier::update_range_type(uint32_t n_samples, uint32_t node_index) {
  if (n_samples >= 2) {
    if (is_full()) {
      NodeClassifier &node = nodes[node_index];
      if(node.memorized()) {
        // Node is memorized
      } else {
        // Node is not memorized. If it has more samples than a memorized node with less samples, we tradeoff memory
        auto it = disposable_nodes.begin();
        uint32_t n_samples_worst, node_index_worst;
        std::tie(n_samples_worst, node_index_worst) = *it;
        if(n_samples > n_samples_worst) {
          nodes[node_index_worst].forget_range();
          nodes[node_index].memorize_range();
        }
      }
    } else {
      NodeClassifier &node = nodes[node_index];
      node.memorize_range();
    }
  }
}


void TreeClassifier::add_to_disposables(uint32_t n_samples, uint32_t node_index) {
  std::pair<uint32_t, uint32_t> e(n_samples, node_index);
  disposable_nodes.insert(e);
  _n_nodes_memorized++;
  _n_nodes_computed--;
}

void TreeClassifier::remove_from_disposables(uint32_t n_samples, uint32_t node_index) {
  std::pair<uint32_t, uint32_t> e(n_samples, node_index);
  disposable_nodes.erase(e);
  _n_nodes_memorized--;
  _n_nodes_computed++;
}


void TreeClassifier::incr_n_samples(uint32_t n_samples, uint32_t node_index) {
  std::pair<uint32_t, uint32_t> e(n_samples, node_index);
  disposable_nodes.erase(e);
  e.first++;
  disposable_nodes.insert(e);
}



float TreeClassifier::compute_split_time(uint32_t node_index, uint32_t sample) {
  NodeClassifier &current_node = node(node_index);
  float y_t = sample_label(sample);
  // Don't split if the node is pure: all labels are equal to the one of y_t
  if (!forest.split_pure() && current_node.is_dirac(y_t)) {
    return 0;
  }

  // Don't split if the number of samples in the node is not large enough
  /*
  uint32_t min_samples_split = forest.min_samples_split();
  if ((min_samples_split > 0) && (current_node.n_samples() < min_samples_split)) {
    return 0;
  }


  // Get maximum number of nodes allowed in the tree
  uint32_t max_nodes = forest.max_nodes();
  // Don't split if there is already too many nodes
  if ((max_nodes > 0) && (_n_nodes >= max_nodes)) {
    return 0;
  }
  */

  // Let's compute the extension of the range of the current node, and its sum
  float extensions_sum, extensions_max;

  const ArrayFloat& x_t = sample_features(sample);
  current_node.compute_range_extension(x_t, intensities, extensions_sum, extensions_max);

  // Get the min extension size
/*
  float min_extension_size = forest.min_extension_size();
  // Don't split if the extension is too small
  if ((min_extension_size > 0) && extensions_max < min_extension_size) {
    return 0;
  }
*/

  // If the sample x_t extends the current range of the node
  if (extensions_sum > 0) {
    // TODO: check that intensity is indeed intensity in the rand.h
    float T = forest.sample_exponential(extensions_sum);
    float time = current_node.time();
    // Splitting time of the node (if splitting occurs)
    float split_time = time + T;
    if (current_node.is_leaf()) {
      // If the node is a leaf we must split it
      return split_time;
    }
    // Otherwise we apply Mondrian process dark magic :)
    // 1. We get the creation time of the childs (left and right is the same)
    float child_time = node(current_node.left()).time();
    // 2. We check if splitting time occurs before child creation time
    // Sample a exponential random variable with intensity
    if (split_time < child_time) {
      // std::cout << "Done with TreeClassifier::compute_split_time returning
      // " << split_time << std::endl;
      return split_time;
    }
  }
  return 0;
}

// Split node at time split_time, using the given feature and threshold
void TreeClassifier::split_node(uint32_t node_index, const float split_time, const float threshold,
                                const uint32_t feature, const bool is_right_extension) {

  // TODO: here we can have a copy of a node with memorized range
  uint32_t left_new = add_node(node_index, split_time);
  uint32_t right_new = add_node(node_index, split_time);
  // left_new and right_new nodes don't use memory yet

  // Let's take again the current node (adding node might lead to re-allocations
  // in the nodes std::vector)
  NodeClassifier &current_node = node(node_index);
  NodeClassifier &left_new_node = node(left_new);
  NodeClassifier &right_new_node = node(right_new);
  // The value of the feature
  if (is_right_extension) {
    // left_new is the same as node_index, excepted for the parent, time and the
    // fact that it's a leaf
    left_new_node = current_node;
    // so we need to put back the correct parent and time
    left_new_node.parent(node_index).time(split_time);
    // right_new doit avoir comme parent node_index
    right_new_node.parent(node_index).time(split_time);
    // We must tell the old childs that they have a new parent, if the current
    // node is not a leaf
    if (!current_node.is_leaf()) {
      node(current_node.left()).parent(left_new);
      node(current_node.right()).parent(left_new);
    }
  } else {
    right_new_node = current_node;
    right_new_node.parent(node_index).time(split_time);
    left_new_node.parent(node_index).time(split_time);
    if (!current_node.is_leaf()) {
      node(current_node.left()).parent(right_new);
      node(current_node.right()).parent(right_new);
    }
  }
  // We update the splitting feature, threshold, and childs of the current index
  current_node.feature(feature).threshold(threshold).left(left_new).right(right_new).is_leaf(false);
}

uint32_t TreeClassifier::get_leaf(const ArrayFloat &x_t) {
  // Find the index of the leaf that contains the sample. Start at the root.
  // Index of the root is 0
  uint32_t index_current_node = 0;
  bool is_leaf = false;
  while (!is_leaf) {
    NodeClassifier &current_node = node(index_current_node);
    is_leaf = current_node.is_leaf();
    if (!is_leaf) {
      uint32_t feature = current_node.feature();
      float threshold = current_node.threshold();
      if (x_t[feature] <= threshold) {
        index_current_node = current_node.left();
      } else {
        index_current_node = current_node.right();
      }
    }
  }
  return index_current_node;
}

// Given a sample point, return the depth of the leaf corresponding to the point
// (including root)
uint32_t TreeClassifier::get_path_depth(const ArrayFloat &x_t) {
  uint32_t index_current_node = node(0).get_child(x_t);
  uint32_t depth = 1;
  while (true) {
    NodeClassifier &current_node = node(index_current_node);
    bool is_leaf = current_node.is_leaf();
    if (is_leaf) {
      return depth;
    } else {
      depth++;
      index_current_node = current_node.get_child(x_t);
    }
  }
}

// Given a sample point, return the path to the leaf corresponding to the point
// (not including root)
void TreeClassifier::get_path(const ArrayFloat &x_t, SArrayUIntPtr path) {
  uint32_t index_current_node = node(0).get_child(x_t);
  uint32_t depth = 0;
  while (true) {
    (*path)[depth] = index_current_node;
    NodeClassifier &current_node = node(index_current_node);
    bool is_leaf = current_node.is_leaf();
    if (is_leaf) {
      return;
    } else {
      depth++;
      index_current_node = current_node.get_child(x_t);
    }
  }
}

void TreeClassifier::go_upwards(uint32_t leaf_index) {
  uint32_t current = leaf_index;
  if (iteration >= 1) {
    while (true) {
      NodeClassifier &current_node = node(current);
      current_node.update_weight_tree();
      if (current == 0) {
        // We arrive at the root
        break;
      }
      // We must update the root node
      current = node(current).parent();
    }
  }
}

void TreeClassifier::update_depth(uint32_t node_index, uint8_t depth) {
  NodeClassifier &current_node = node(node_index);
  depth++;
  current_node.depth(depth);
  if (current_node.is_leaf()) {
    return;
  } else {
    update_depth(current_node.left(), depth);
    update_depth(current_node.right(), depth);
  }
}

void TreeClassifier::print() {
  std::cout << "Tree(n_nodes: " << _n_nodes << "," << std::endl;
  for (uint32_t node_index = 0; node_index < _n_nodes; ++node_index) {
    std::cout << " ";
    std::cout << "index: " << node_index << " ";
    nodes[node_index].print();
  }
  std::cout << ")" << std::endl;
}

void TreeClassifier::predict(const ArrayFloat &x_t, ArrayFloat &scores, bool use_aggregation) {
  uint32_t leaf = get_leaf(x_t);
  if (!use_aggregation) {
    node(leaf).predict(scores);
    return;
  }
  uint32_t current = leaf;
  // The child of the current node that does not contain the data
  ArrayFloat pred_new(n_classes());
  while (true) {
    NodeClassifier &current_node = node(current);
    if (current_node.is_leaf()) {
      current_node.predict(scores);
    } else {
      float w = std::exp(current_node.weight() - current_node.weight_tree());
      // Get the predictions of the current node
      current_node.predict(pred_new);
      for (uint8_t c = 0; c < n_classes(); ++c) {
        scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c];
      }
    }
    // Root must be updated as well
    if (current == 0) {
      break;
    }
    current = current_node.parent();
  }
}

void TreeClassifier::reserve_nodes(uint32_t n_nodes) {
  nodes.reserve(n_nodes);
  // TODO: requires some thought...
  for (uint32_t i = 0; i < n_nodes; ++i) {
    nodes.emplace_back(*this, 0, 0, 0);
  }
}

void TreeClassifier::create_root() {
  nodes.emplace_back(*this, 0, 0, 0);
  _n_nodes++;
  _n_nodes_computed++;
}

uint32_t TreeClassifier::add_node(uint32_t parent, float time) {
  if (_n_nodes < nodes.size()) {
    // We have enough nodes already, so let's use the last free one, and just
    // update its time and parent
    node(_n_nodes).index(_n_nodes).parent(parent).time(time);
    _n_nodes++;
    // This node is brand new, so it's necessary computed
    _n_nodes_computed++;
    return _n_nodes - 1;
  } else {
    TICK_ERROR("OnlineForest: Something went wrong with nodes allocation !!!")
  }
}

inline const ArrayFloat& TreeClassifier::sample_features(uint32_t sample) const {
  return forest.sample_features(sample);
}

inline float TreeClassifier::sample_label(uint32_t sample) const {
  return forest.sample_label(sample);
}

inline uint32_t TreeClassifier::n_features() const { return _n_features; }

inline uint8_t TreeClassifier::n_classes() const { return _n_classes; }

inline uint32_t TreeClassifier::n_nodes() const { return _n_nodes; }

inline uint32_t TreeClassifier::n_nodes_reserved() const {
  return static_cast<uint32_t>(nodes.size());
}

inline uint32_t TreeClassifier::max_nodes_with_memory() const {
  return forest.max_nodes_with_memory();
}

uint32_t TreeClassifier::n_leaves() const {
  uint32_t n_leaves = 0;
  for (uint32_t node_index = 0; node_index < _n_nodes; ++node_index) {
    const NodeClassifier &node = nodes[node_index];
    if (node.is_leaf()) {
      ++n_leaves;
    }
  }
  return n_leaves;
}

inline float TreeClassifier::step() const { return forest.step(); }

inline float TreeClassifier::dirichlet() const { return forest.dirichlet(); }

inline CriterionClassifier TreeClassifier::criterion() const { return forest.criterion(); }

inline bool TreeClassifier::use_aggregation() const { return forest.use_aggregation(); }

FeatureImportanceType TreeClassifier::feature_importance_type() const {
  return forest.feature_importance_type();
}

inline float TreeClassifier::feature_importance(const uint32_t j) const {
  if (feature_importance_type() == FeatureImportanceType::no) {
    return 1;
  } else {
    if (feature_importance_type() == FeatureImportanceType::estimated) {
      return feature_importances_[j];
    } else {
      return (iteration + 1) * given_feature_importance(j);
    }
  }
}

float TreeClassifier::given_feature_importance(const uint32_t j) const {
  return forest.given_feature_importances(j);
}

void TreeClassifier::get_flat_nodes(
    SArrayUIntPtr nodes_parent, SArrayUIntPtr nodes_left, SArrayUIntPtr nodes_right,
    SArrayUIntPtr nodes_feature, SArrayFloatPtr nodes_threshold, SArrayFloatPtr nodes_time,
    SArrayUShortPtr nodes_depth, SArrayFloat2dPtr nodes_features_min,
    SArrayFloat2dPtr nodes_features_max, SArrayUIntPtr nodes_n_samples, SArrayUIntPtr nodes_sample,
    SArrayFloatPtr nodes_weight,
    SArrayFloatPtr nodes_weight_tree, SArrayUShortPtr nodes_is_leaf, SArrayUShortPtr nodes_is_memorized, SArrayUInt2dPtr nodes_counts) {
  for (uint32_t node_index = 0; node_index < _n_nodes; ++node_index) {
    NodeClassifier &node = nodes[node_index];
    (*nodes_parent)[node_index] = node.parent();
    (*nodes_left)[node_index] = node.left();
    (*nodes_right)[node_index] = node.right();
    (*nodes_feature)[node_index] = node.feature();
    (*nodes_threshold)[node_index] = node.threshold();
    (*nodes_time)[node_index] = node.time();
    (*nodes_depth)[node_index] = node.depth();
    (*nodes_n_samples)[node_index] = node.n_samples();
    (*nodes_weight)[node_index] = node.weight();
    (*nodes_weight_tree)[node_index] = node.weight_tree();
    // nodes_is_leaf->operator[](node_index) = static_cast<ushort>(node.is_leaf());
    (*nodes_is_leaf)[node_index] = static_cast<ushort>(node.is_leaf());
    (*nodes_is_memorized)[node_index] = static_cast<ushort>(node.memorized());

    ArrayFloat features_min = view_row(*nodes_features_min, node_index);
    ArrayFloat features_max = view_row(*nodes_features_max, node_index);
    for(uint32_t j = 0; j < _n_features; ++j) {
      float range_min, range_max;
      std::tie(range_min, range_max) = node.range(j);
      // features_min[j] = node.features_min(j);
      // features_max[j] = node.features_max(j);
      features_min[j] = range_min;
      features_max[j] = range_max;
    }
  }
}


void TreeClassifier::inspect_nodes_memory() const {
  uint32_t n_memory = 0;
  uint32_t n_computed = 0;
  for (uint32_t node_index = 0; node_index < _n_nodes; ++node_index) {
    const NodeClassifier &node = nodes[node_index];
    if (node.memorized()) {
      n_memory++;
    } else {
      n_computed++;
    }
  }
  std::cout << "computed: " << n_computed << ", memorized: " << n_memory << std::endl;
  std::cout << "computed: " <<_n_nodes_computed << ", memorized: " << _n_nodes_memorized << std::endl;
  std::cout << "disposable_nodes.size(): " << disposable_nodes.size();
  std::cout << ", max_nodes_with_memory: " << max_nodes_with_memory();
  std::cout << ", n_nodes: " << _n_nodes << std::endl;

  /*
  std::cout << "disposable_nodes: ";
  for (auto it = disposable_nodes.begin(); it != disposable_nodes.end(); ++it) {
    std::cout << "(" << (*it).first << ", " << (*it).second << "), ";
  }
  std::cout << std::endl;
  */
}


/*********************************************************************************
 * OnlineForestClassifier methods
 *********************************************************************************/

// TODO: remove n_passes and subsampling
// TODO: Add the bootstrap option ? (supposedly useless)
// TODO: add a compute_depth option

OnlineForestClassifier::OnlineForestClassifier(
    uint32_t n_features, uint8_t n_classes, uint8_t n_trees, float step,
    CriterionClassifier criterion, FeatureImportanceType feature_importance_type,
    bool use_aggregation, float dirichlet, bool split_pure, int32_t max_nodes,
    float min_extension_size, int32_t min_samples_split, int32_t max_features, int32_t n_threads,
    int seed, bool verbose, uint32_t print_every, uint32_t max_nodes_with_memory)
    : _n_samples(0), _n_features(n_features),
      _n_classes(n_classes),
      _n_trees(n_trees),
      _step(step),
      _criterion(criterion),
      _feature_importance_type(feature_importance_type),
      _use_aggregation(use_aggregation),
      _dirichlet(dirichlet),
      _split_pure(split_pure),
      _max_nodes(max_nodes),
      _min_extension_size(min_extension_size),
      _min_samples_split(min_samples_split),
      _max_features(max_features),
      _n_threads(n_threads),
      _verbose(verbose),
      _print_every(print_every),
      _max_nodes_with_memory(max_nodes_with_memory),
      rand(seed) {
  // No iteration so far
  _iteration = 0;

  create_trees();
  _samples_features = std::vector<ArrayFloat>();
  _samples_label = std::vector<float>();
}

OnlineForestClassifier::~OnlineForestClassifier() {}

void OnlineForestClassifier::create_trees() {
  // Just in case...
  trees.clear();
  trees.reserve(_n_trees);
  // Better tree allocation
  for (uint32_t i = 0; i < _n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

// features_start: beginning of the features vector to add
void OnlineForestClassifier::add_sample(float *features_start, float label) {
  // Copy the features vector of the sample
  _samples_features.emplace_back(ArrayFloat(_n_features));
  std::copy(features_start, features_start + _n_features, _samples_features.back().data());
  // Copy the label
  _samples_label.emplace_back(label);
  _n_samples++;
}

void OnlineForestClassifier::fit(const SArrayFloat2dPtr features, const SArrayFloatPtr labels) {
  uint32_t n_samples_batch = static_cast<uint32_t>(features->n_rows());
  uint32_t n_features = static_cast<uint32_t>(features->n_cols());
  // Check or initialize number of features
  check_n_features(n_features, false);

  // Pre-allocate space for the nodes in all them trees
  for (TreeClassifier &tree : trees) {
    // Maximum number of nodes is now the current one + number of samples in
    // this batch
    // TODO: IMPORTANT !!!! are we sure about this ?????
    // TODO: tree.reserve_nodes(tree.n_nodes() + 2 * n_samples_batch - 1);
    // tree.reserve_nodes(2 * tree.n_nodes() + 2 * n_samples_batch);
    if (tree.n_nodes() + 2 * n_samples_batch > tree.n_nodes_reserved()) {
      tree.reserve_nodes(tree.n_nodes() + 2 * n_samples_batch);
      tree.reserve_nodes(tree.n_nodes() + 2 * n_samples_batch);
    }
  }

  for (uint32_t i = 0; i < n_samples_batch; ++i) {
    // Get pointer to the beginning of the ith features vector
    ArrayFloat features_i = view_row(*features, i);
    float *features_start = features_i.data();
    // Get the label of the ith sample and check it
    float label = (*labels)[i];
    check_label(label);
    // We add this sample (features, label) pair to the forest
    add_sample(features_start, label);
    // Fit the trees using this sample
    for (TreeClassifier &tree : trees) {
      tree.fit(last_sample());
      // tree.print();
    }
    _iteration++;
    if(_verbose) {
      if (_iteration % _print_every == 0) {
        std::cout << "iteration: " << _iteration << std::endl;
      }
    }
  }
}

void OnlineForestClassifier::predict(const SArrayFloat2dPtr features, SArrayFloat2dPtr scores) {
  scores->fill(0.);
  uint32_t n_features = static_cast<uint32_t>(features->n_cols());
  check_n_features(n_features, true);
  if (_iteration > 0) {
    uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
    ArrayFloat scores_tree(_n_classes);
    scores_tree.fill(0.);
    ArrayFloat scores_forest(_n_classes);
    scores_forest.fill(0.);
    for (uint32_t i = 0; i < n_samples; ++i) {
      // The prediction is simply the average of the predictions
      ArrayFloat scores_i = view_row(*scores, i);
      for (TreeClassifier &tree : trees) {
        tree.predict(view_row(*features, i), scores_tree, _use_aggregation);
        // TODO: use a .incr method instead ??
        scores_i.mult_incr(scores_tree, 1.);
      }
      scores_i /= _n_trees;
    }
  } else {
    TICK_ERROR("You must call ``fit`` before ``predict``.")
  }
}

void OnlineForestClassifier::clear() {
  create_trees();
  _samples_features.clear();
  _samples_label.clear();
  _iteration = 0;
  _n_samples = 0;
}

void OnlineForestClassifier::print() {
  for (TreeClassifier &tree : trees) {
    tree.print();
  }
}

inline float OnlineForestClassifier::sample_exponential(float intensity) {
  return rand.exponential(intensity);
}

inline uint32_t OnlineForestClassifier::sample_feature(const ArrayFloat &prob) {
  // TODO: warning this assumes that prob is indeed a probability vector... !
  return rand.discrete(prob);
}

inline float OnlineForestClassifier::sample_threshold(float left, float right) {
  return rand.uniform(left, right);
}

void OnlineForestClassifier::n_nodes(SArrayUIntPtr n_nodes_per_tree) {
  uint8_t j = 0;
  for (TreeClassifier &tree : trees) {
    (*n_nodes_per_tree)[j] = tree.n_nodes();
    j++;
  }
}

void OnlineForestClassifier::n_nodes_reserved(SArrayUIntPtr n_reserved_nodes_per_tree) {
  uint8_t j = 0;
  for (TreeClassifier &tree : trees) {
    (*n_reserved_nodes_per_tree)[j] = tree.n_nodes_reserved();
    j++;
  }
}

void OnlineForestClassifier::n_leaves(SArrayUIntPtr n_leaves_per_tree) {
  uint8_t j = 0;
  for (TreeClassifier &tree : trees) {
    (*n_leaves_per_tree)[j] = tree.n_leaves();
    j++;
  }
}

bool OnlineForestClassifier::verbose() const { return _verbose; }
OnlineForestClassifier &OnlineForestClassifier::verbose(bool verbose) {
  _verbose = verbose;
  return *this;
}

bool OnlineForestClassifier::use_aggregation() const { return _use_aggregation; }

float OnlineForestClassifier::step() const { return _step; }
OnlineForestClassifier &OnlineForestClassifier::step(const float step) {
  _step = step;
  return *this;
}

uint32_t OnlineForestClassifier::n_samples() const {
  return _n_samples;
}

uint32_t OnlineForestClassifier::n_features() const {
  return _n_features;
}

uint32_t OnlineForestClassifier::last_sample() const {
  // TODO: raise error is _n_sample
  if (_n_samples >= 1) {
    return _n_samples - 1;
  } else {
    TICK_ERROR("No last() sample in the forest: none have been inserted!");
  }
}

void OnlineForestClassifier::check_n_features(uint32_t n_features, bool predict) {
  if (_iteration == 0) {
    _n_features = n_features;
  } else {
    if (n_features != _n_features) {
      if (predict) {
        TICK_ERROR("Wrong number of features: trained with " + std::to_string(_n_features) +
            " features, but received " + std::to_string(n_features) +
            " features for prediction");
      } else {
        TICK_ERROR("Wrong number of features: started to train with " + std::to_string(_n_features) +
            " features, but received " + std::to_string(n_features) + " afterwards");
      }
    }
  }
}

void OnlineForestClassifier::check_label(double label) const {
  double iptr;
  double fptr = std::modf(label, &iptr);
  if (fptr != 0) {
    TICK_ERROR("Wrong label type: received " + std::to_string(label) +
               " for a classification problem");
  }
  if ((label < 0) || (label >= _n_classes)) {
    TICK_ERROR("Wrong label value: received " + std::to_string(label) +
               " while training for classification with " + std::to_string(_n_classes) +
               " classes.");
  }
}

uint8_t OnlineForestClassifier::n_classes() const { return _n_classes; }

uint8_t OnlineForestClassifier::n_trees() const { return _n_trees; }

float OnlineForestClassifier::given_feature_importances(const ulong j) const {
  return _given_feature_importances[j];
}

int32_t OnlineForestClassifier::n_threads() const { return _n_threads; }

CriterionClassifier OnlineForestClassifier::criterion() const { return _criterion; }

int OnlineForestClassifier::seed() const { return _seed; }

OnlineForestClassifier &OnlineForestClassifier::seed(int seed) {
  _seed = seed;
  rand.reseed(seed);
  return *this;
}

OnlineForestClassifier &OnlineForestClassifier::n_threads(int32_t n_threads) {
  _n_threads = n_threads;
  return *this;
}

OnlineForestClassifier &OnlineForestClassifier::criterion(CriterionClassifier criterion) {
  _criterion = criterion;
  return *this;
}

FeatureImportanceType OnlineForestClassifier::feature_importance_type() const {
  return _feature_importance_type;
}

OnlineForestClassifier &OnlineForestClassifier::given_feature_importances(
    const ArrayFloat &feature_importances) {
  for (ulong j = 0; j < n_features(); ++j) {
    _given_feature_importances[j] = static_cast<float>(feature_importances[j]);
  }
  return *this;
}

float OnlineForestClassifier::dirichlet() const { return _dirichlet; }

OnlineForestClassifier &OnlineForestClassifier::dirichlet(const float dirichlet) {
  // TODO: check that it's > 0
  _dirichlet = dirichlet;
  return *this;
}

uint32_t OnlineForestClassifier::max_nodes_with_memory() const { return _max_nodes_with_memory; }

bool OnlineForestClassifier::split_pure() const { return _split_pure; }

int32_t OnlineForestClassifier::max_nodes() const { return _max_nodes; }

float OnlineForestClassifier::min_extension_size() const { return _min_extension_size; }

int32_t OnlineForestClassifier::min_samples_split() const { return _min_samples_split; }

int32_t OnlineForestClassifier::max_features() const { return _max_features; }

const ArrayFloat& OnlineForestClassifier::sample_features(uint32_t sample) const {
  if (sample >= _n_samples) {
    TICK_ERROR("Asking for sample " + std::to_string(sample) + " but I have only seen "
                   + std::to_string(_n_samples) + " samples so far");
  } else {
    return _samples_features[sample];
  }
}

float OnlineForestClassifier::sample_label(uint32_t sample) const {
  if (sample >= _n_samples) {
    TICK_ERROR("Asking for sample " + std::to_string(sample) + " but I have only seen "
                   + std::to_string(_n_samples) + " samples so far");
  } else {
    return _samples_label[sample];
  }
}

void OnlineForestClassifier::get_feature_importances(SArrayFloatPtr feature_importances) {
  if (_feature_importance_type == FeatureImportanceType::estimated) {
    const float a = static_cast<float>(1) / n_trees();
    ArrayFloat importances(_n_features);
    importances.fill(0);
    for (TreeClassifier &tree : trees) {
      importances.mult_incr(tree.feature_importances(), a);
    }
    importances /= (importances.sum());
    for (ulong j = 0; j < _n_features; ++j) {
      (*feature_importances)[j] = importances[j];
    }
  } else {
    if (_feature_importance_type == FeatureImportanceType::given) {
      for (ulong j = 0; j < _n_features; ++j) {
        (*feature_importances)[j] = _given_feature_importances[j];
      }
    }
  }
}

uint32_t OnlineForestClassifier::get_path_depth(const uint8_t tree, const SArrayFloatPtr x_t) {
  return trees[tree].get_path_depth(*x_t);
}

// Get the path of x_t
void OnlineForestClassifier::get_path(const uint8_t tree, const SArrayFloatPtr x_t,
                                      SArrayUIntPtr path) {
  trees[tree].get_path(*x_t, path);
}

void OnlineForestClassifier::get_flat_nodes(
    uint8_t tree,
    SArrayUIntPtr nodes_parent,
    SArrayUIntPtr nodes_left,
    SArrayUIntPtr nodes_right,
    SArrayUIntPtr nodes_feature,
    SArrayFloatPtr nodes_threshold,
    SArrayFloatPtr nodes_time,
    SArrayUShortPtr nodes_depth,
    SArrayFloat2dPtr nodes_features_min,
    SArrayFloat2dPtr nodes_features_max,
    SArrayUIntPtr nodes_n_samples,
    SArrayUIntPtr nodes_sample,
    SArrayFloatPtr nodes_weight,
    SArrayFloatPtr nodes_weight_tree,
    SArrayUShortPtr nodes_is_leaf,
    SArrayUShortPtr nodes_is_memorized,
    SArrayUInt2dPtr nodes_counts) {
  trees[tree].get_flat_nodes(
      nodes_parent,
      nodes_left,
      nodes_right,
      nodes_feature,
      nodes_threshold,
      nodes_time,
      nodes_depth,
      nodes_features_min,
      nodes_features_max,
      nodes_n_samples,
      nodes_sample,
      nodes_weight,
      nodes_weight_tree,
      nodes_is_leaf,
      nodes_is_memorized,
      nodes_counts);
}


