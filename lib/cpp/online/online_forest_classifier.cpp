// License: BSD 3 clause

#include "tick/online/online_forest_classifier.h"

/*********************************************************************************
 * NodeClassifier methods
 *********************************************************************************/

NodeClassifier::NodeClassifier(TreeClassifier &tree, uint32_t parent, float time)
    : _tree(tree), _parent(parent), _left(0), _right(0), _time(time), _depth(0),
      _features_min(tree.n_features()), _features_max(tree.n_features()),
      _n_samples(0), _is_leaf(true), _weight(0), _weight_tree(0),
      _counts(tree.n_classes()) {
  _counts.fill(0);
}

NodeClassifier::NodeClassifier(const NodeClassifier &node)
    : _tree(node._tree), _parent(node._parent), _left(node._left), _right(node._right),
      _feature(node._feature), _threshold(node._threshold),
      _time(node._time), _depth(node._depth), _features_min(node._features_min), _features_max(node._features_max),
      _n_samples(node._n_samples),
      _y_t(node._y_t),
      _weight(node._weight), _weight_tree(node._weight_tree),
      _is_leaf(node._is_leaf),
      _counts(node._counts) {}

NodeClassifier::NodeClassifier(const NodeClassifier &&node) : _tree(_tree) {
  _parent = node._parent;
  _left = node._left;
  _right = node._right;
  _feature = node._feature;
  _threshold = node._threshold;
  _time = node._time;
  _depth = node._depth;
  _features_min = node._features_min;
  _features_max = node._features_max;
  _n_samples = node._n_samples;
  _y_t = node._y_t;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _counts = node._counts;
}

NodeClassifier &NodeClassifier::operator=(const NodeClassifier &node) {
  _parent = node._parent;
  _left = node._left;
  _right = node._right;
  _feature = node._feature;
  _threshold = node._threshold;
  _time = node._time;
  _depth = node._depth;
  _features_min = node._features_min;
  _features_max = node._features_max;
  _n_samples = node._n_samples;
  _y_t = node._y_t;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _counts = node._counts;
  return *this;
}

void NodeClassifier::update_range(const ArrayDouble &x_t) {
  if (_n_samples == 0) {
    for (ulong j = 0; j < n_features(); ++j) {
      float x_tj = static_cast<float>(x_t[j]);
      _features_min[j] = x_tj;
      _features_max[j] = x_tj;
    }
  } else {
    for (ulong j = 0; j < n_features(); ++j) {
      float x_tj = static_cast<float>(x_t[j]);
      if (x_tj < _features_min[j]) {
        _features_min[j] = x_tj;
      }
      if (x_tj > _features_max[j]) {
        _features_max[j] = x_tj;
      }
    }
  }
}

void NodeClassifier::update_n_samples() {
  _n_samples++;
}

float NodeClassifier::update_weight(const ArrayDouble &x_t, const double y_t) {
  float loss_t = loss(y_t);
  if (use_aggregation()) {
    _weight -= step() * loss_t;
  }
  // We return the loss before updating the predictor of the node in order to
  // update the feature importance in TreeClassifier::go_downwards
  return loss_t;
}

void NodeClassifier::update_count(const double y_t) {
  // We update the counts for the class y_t
  _counts[static_cast<uint8_t>(y_t)]++;
}

void NodeClassifier::update_downwards(const ArrayDouble &x_t, const double y_t, bool do_update_weight) {
  update_range(x_t);
  update_n_samples();
  if (do_update_weight) {
    update_weight(x_t, y_t);
  }
  update_count(y_t);
}

void NodeClassifier::update_weight_tree() {
  if (_is_leaf) {
    _weight_tree = _weight;
  } else {
    _weight_tree = log_sum_2_exp(_weight, node(_left).weight_tree() + node(_right).weight_tree());
  }
}

bool NodeClassifier::is_dirac(const double y_t) {
  // Returns true if the node only has the same label as y_t
  return (_n_samples == _counts[static_cast<uint8_t>(y_t)]);
}

uint32_t NodeClassifier::get_child(const ArrayDouble &x_t) {
  if (static_cast<float>(x_t[_feature]) <= _threshold) {
    return _left;
  } else {
    return _right;
  }
}

float NodeClassifier::score(uint8_t c) const {
  // Using the Dirichet prior
  return (_counts[c] + dirichlet()) / (_n_samples + dirichlet() * n_classes());
}

float NodeClassifier::compute_range_extension(const ArrayDouble &x_t, ArrayFloat &extensions) {
  float sum_extensions = 0;
  for (uint32_t j = 0; j < n_features(); ++j) {
    float x_tj = static_cast<float>(x_t[j]);
    float feature_min_j = _features_min[j];
    float feature_max_j = _features_max[j];
    if (x_tj < feature_min_j) {
      float diff = feature_min_j - x_tj;
      extensions[j] = diff;
      sum_extensions += diff;
    } else {
      if (x_tj > feature_max_j) {
        float diff = x_tj - feature_max_j;
        extensions[j] = diff;
        sum_extensions += diff;
      } else {
        extensions[j] = 0;
      }
    }
  }
  return sum_extensions;
}

void NodeClassifier::predict(ArrayDouble &scores) const {
  for (uint8_t c = 0; c < n_classes(); ++c) {
    scores[c] = static_cast<double>(score(c));
  }
}

float NodeClassifier::loss(const double y_t) {
  // Log-loss
  uint8_t c = static_cast<uint8_t>(y_t);
  return -std::log(score(c));
}

inline NodeClassifier &NodeClassifier::node(uint32_t index) const {
  return _tree.node(index);
}

uint32_t NodeClassifier::n_features() const {
  return _tree.n_features();
}

uint8_t NodeClassifier::n_classes() const {
  return _tree.n_classes();
}

inline float NodeClassifier::step() const {
  return _tree.step();
}

inline float NodeClassifier::dirichlet() const {
  return _tree.dirichlet();
}

inline uint32_t NodeClassifier::parent() const {
  return _parent;
}

inline NodeClassifier &NodeClassifier::set_parent(uint32_t parent) {
  _parent = parent;
  return *this;
}

inline uint32_t NodeClassifier::left() const {
  return _left;
}

inline NodeClassifier &NodeClassifier::set_left(uint32_t left) {
  _left = left;
  return *this;
}

inline uint32_t NodeClassifier::right() const {
  return _right;
}

inline NodeClassifier &NodeClassifier::set_right(uint32_t right) {
  _right = right;
  return *this;
}

inline bool NodeClassifier::is_leaf() const {
  return _is_leaf;
}

inline NodeClassifier &NodeClassifier::set_is_leaf(bool is_leaf) {
  _is_leaf = is_leaf;
  return *this;
}

inline uint32_t NodeClassifier::feature() const {
  return _feature;
}

inline NodeClassifier &NodeClassifier::set_feature(uint32_t feature) {
  _feature = feature;
  return *this;
}

inline float NodeClassifier::threshold() const {
  return _threshold;
}

inline NodeClassifier &NodeClassifier::set_threshold(float threshold) {
  _threshold = threshold;
  return *this;
}

inline float NodeClassifier::time() const {
  return _time;
}

inline NodeClassifier &NodeClassifier::set_time(float time) {
  _time = time;
  return *this;
}

inline uint8_t NodeClassifier::depth() const {
  return _depth;
}

inline NodeClassifier &NodeClassifier::set_depth(uint8_t depth) {
  _depth = depth;
  return *this;
}

inline float NodeClassifier::features_min(const uint32_t j) const {
  return _features_min[j];
}

inline float NodeClassifier::features_max(const uint32_t j) const {
  return _features_max[j];
}

inline uint32_t NodeClassifier::n_samples() const {
  return _n_samples;
}

inline bool NodeClassifier::use_aggregation() const {
  return _tree.use_aggregation();
}

inline float NodeClassifier::weight() const {
  return _weight;
}

inline float NodeClassifier::weight_tree() const {
  return _weight_tree;
}

void NodeClassifier::print() {
  std::cout << "Node(parent: " << _parent
            << ", left: " << _left
            << ", right: " << _right
            << ", time: " << std::setprecision(2) << _time
            << ", n_samples: " << _n_samples
            << ", is_leaf: " << _is_leaf
            // << ", feature: " << _feature
            // << ", thresh: " << _threshold
            << ", scores: [" << std::setprecision(2) << score(0) << ", " << std::setprecision(2) << score(1) << "]"
            << ", counts: [" << std::setprecision(2) << _counts[0] << ", " << std::setprecision(2) << _counts[1] << "]";
//  if (_n_samples > 0) {
//    std::cout << ", min: [" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
//              << _features_min[1] << "]"
//              << ", max: [" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
//              << _features_max[1] << "]";
//  }
//  std::cout << ", weight: " << _weight
//            << ", weight_tree: " << _weight_tree
//            << ")\n";
  std::cout << ")\n";
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

void TreeClassifier::fit(const ArrayDouble &x_t, double y_t) {
//  std::cout << "++++++++++++++++++++++++++++++++" << std::endl;
//  std::cout << "Starting iteration: " << iteration << std::endl;
  uint32_t leaf = go_downwards(x_t, y_t);

  if (use_aggregation()) {
    go_upwards(leaf);
  }
//  std::cout << "Finished iteration: " << iteration << std::endl;
//  print();
  iteration++;
}

uint32_t TreeClassifier::go_downwards(const ArrayDouble &x_t, double y_t) {
  // We update the nodes along the path which leads to the leaf containing x_t
  // For each node on the path, we consider the possibility of splitting it,
  // following the Mondrian process definition.
  // Index of the root is 0
  uint32_t index_current_node = 0;
  bool is_leaf = false;
  float loss_t;

  if (iteration == 0) {
    // If it's the first iteration, we just put the point in the range of root
    // Let's get the root
    NodeClassifier &current_node = node(0);
    current_node.update_downwards(x_t, y_t, false);
    // current_node.update_range(x_t);
    // current_node.update_n_samples();
    // And we update the label count of root
    // current_node.update_count(y_t);
    return index_current_node;
  } else {
    while (true) {
      // If it's not the first iteration (otherwise the current node is root
      // with no range), we consider the possibility of a split
      float split_time = compute_split_time(index_current_node, x_t);
//      std::cout << "iteration: " << iteration;
//      std::cout << ", split_time: " << split_time << std::endl;

      if (split_time > 0) {
        NodeClassifier &current_node = node(index_current_node);
        // We split the current node: because the current node is a leaf, or because we add a new node along the path
        // std::cout << "current node before split: " << index_current_node << std::endl;
        // current_node.print();
        // Normalize the range extensions to get probabilities
        intensities /= intensities.sum();
        // Sample the feature at random with with a probability proportional to the range extensions
        uint32_t feature = forest.sample_feature(intensities);
        // Value of the feature
        float x_tf = static_cast<float>(x_t[feature]);
        // Is it a right extension of the node ?
        bool is_right_extension = x_tf > current_node.features_max(feature);
        float threshold;
        if (is_right_extension) {
          threshold = forest.sample_threshold(current_node.features_max(feature), x_tf);
        } else {
          threshold = forest.sample_threshold(x_tf, current_node.features_min(feature));
        }
//        std::cout << "node: " << index_current_node;
//        std::cout << ", split_time: " << split_time;
//        std::cout << ", threshold: " << threshold;
//        std::cout << ", feature: " << feature;
//        std::cout << ", is_right_extension: " << is_right_extension;
//        std::cout << std::endl;

        split_node(index_current_node, split_time, threshold, feature, is_right_extension);

        // Update the depth of the new childs of the current node

        // Get the current node again, and update it
        NodeClassifier &current_node_again = node(index_current_node);
        current_node_again.update_downwards(x_t, y_t, true);

        // current_node_again.update_range(x_t);
        // current_node_again.update_n_samples();
        // First update the weights, then update the counts...
        // loss_t = current_node_again.update_weight(x_t, y_t);
        // current_node_again.update_count(y_t);
//        std::cout << "current node after split: " << index_current_node << std::endl;
//        current_node_again.print();
//        nodes[index_current_node].print();

        uint32_t left = current_node_again.left();
        uint32_t right = current_node_again.right();
        uint8_t depth = current_node_again.depth();

        // Now, get the next node
        if (is_right_extension) {
          index_current_node = right;
        } else {
          index_current_node = left;
        }

        update_depth(left, depth);
        update_depth(right, depth);

        // This is the leaf containing the sample point (we've just splitted the current node with the data point)
        NodeClassifier &leaf = node(index_current_node);
        // Let's update the leaf containing the point
        leaf.update_downwards(x_t, y_t, false);
        // leaf.update_range(x_t);
        // leaf.update_n_samples();
        // leaf.update_count(y_t);
        // We don't update the weight of the leaf, since we have a single point... and we stop
        // std::cout << "New leaf node: " << index_current_node << std::endl;
        // leaf.print();
        // nodes[index_current_node].print();
        return index_current_node;
      } else {
        // There is no split, so we just update the node and go to the next one
        NodeClassifier &current_node = node(index_current_node);
        current_node.update_downwards(x_t, y_t, true);
        // current_node.update_range(x_t);
        // current_node.update_n_samples();
        // loss_t = current_node.update_weight(x_t, y_t);
        // current_node.update_count(y_t);
        // std::cout << "current node with no split: " << index_current_node << std::endl;
        // current_node.print();
        // nodes[index_current_node].print();
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

float TreeClassifier::compute_split_time(uint32_t node_index, const ArrayDouble &x_t) {
  // std::cout << "TreeClassifier::compute_split_time with node_index: " << node_index << std::endl;
  NodeClassifier &current_node = node(node_index);
  // Let's compute the extension of the range of the current node, and its sum
  float intensities_sum = current_node.compute_range_extension(x_t, intensities);
  // TODO: Make this a parameter
  float min_extension_size = 1e-5;
  // If the sample x_t extends the current range of the node
  if (intensities_sum > min_extension_size) {
    // std::cout << "intensities_sum: " << intensities_sum;
    // TODO: check that intensity is indeed intensity in the rand.h
    float T = forest.sample_exponential(intensities_sum);
    float time = current_node.time();
    // Splitting time of the node (if splitting occurs)
    float split_time = time + T;
    // std::cout << ", T: " << T << ", time: " << time << ", split_time: " << split_time;
    if (current_node.is_leaf()) {
      // If the node is a leaf we must split it
      // std::cout << std::endl;
      // std::cout << "Done with TreeClassifier::compute_split_time returning " << split_time << std::endl;
      return split_time;
    } else {
      // Otherwise we apply Mondrian process dark magic :)
      // 1. We get the creation time of the childs (left and right is the same)
      float child_time = node(current_node.left()).time();
      // std::cout << ", child_time: " << child_time << std::endl;
      // 2. We check if splitting time occurs before child creation time
      // Sample a exponential random variable with intensity
      if (split_time < child_time) {
        // std::cout << "Done with TreeClassifier::compute_split_time returning " << split_time << std::endl;
        return split_time;
      } else {
        // std::cout << "Done with TreeClassifier::compute_split_time returning " << 0 << std::endl;
        return 0;
      }
    }
  } else {
    // std::cout << "Done with TreeClassifier::compute_split_time returning " << 0 << std::endl;
    return 0;
  }
}

// Split node at time split_time, using the given feature and threshold
void TreeClassifier::split_node(uint32_t node_index,
                                const float split_time,
                                const float threshold,
                                const uint32_t feature,
                                const bool is_right_extension) {
  // std::cout << "TreeClassifier::split_node with node_index: " << node_index << std::endl;
  // Get new nodes
  uint32_t left_new = add_node(node_index, split_time);
  uint32_t right_new = add_node(node_index, split_time);
  // Let's take again the current node (adding node might lead to re-allocations in the nodes std::vector)
  NodeClassifier &current_node = node(node_index);
  NodeClassifier &left_new_node = node(left_new);
  NodeClassifier &right_new_node = node(right_new);
  // The value of the feature
  if (is_right_extension) {
    // left_new is the same as node_index, excepted for the parent, time and the fact that it's a leaf
    left_new_node = current_node;
    // so we need to put back the correct parent and time
    left_new_node.set_parent(node_index).set_time(split_time);
    // right_new doit avoir comme parent node_index
    right_new_node.set_parent(node_index).set_time(split_time);
    // We must tell the old childs that they have a new parent, if the current node is not a leaf
    if (!current_node.is_leaf()) {
      node(current_node.left()).set_parent(left_new);
      node(current_node.right()).set_parent(left_new);
    }
  } else {
    right_new_node = current_node;
    right_new_node.set_parent(node_index).set_time(split_time);
    left_new_node.set_parent(node_index).set_time(split_time);
    if (!current_node.is_leaf()) {
      node(current_node.left()).set_parent(right_new);
      node(current_node.right()).set_parent(right_new);
    }
  }
  // We update the splitting feature, threshold, and childs of the current index
  current_node.set_feature(feature).set_threshold(threshold).set_left(left_new)
      .set_right(right_new).set_is_leaf(false);
  // std::cout << "Done with TreeClassifier::split_node." << std::endl;
}

uint32_t TreeClassifier::get_leaf(const ArrayDouble &x_t) {
  // Find the index of the leaf that contains the sample.
  // Start at the root. Index of the root is 0
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

// Given a sample point, return the depth of the leaf corresponding to the point (including root)
uint32_t TreeClassifier::get_path_depth(const ArrayDouble &x_t) {
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

// Given a sample point, return the path to the leaf corresponding to the point (not including root)
void TreeClassifier::get_path(const ArrayDouble &x_t, SArrayUIntPtr path) {
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

  NodeClassifier & current_node = node(node_index);
  depth++;
  current_node.set_depth(depth);
  if(current_node.is_leaf()) {
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


void TreeClassifier::predict(const ArrayDouble &x_t, ArrayDouble &scores, bool use_aggregation) {
  uint32_t leaf = get_leaf(x_t);
  if (!use_aggregation) {
    node(leaf).predict(scores);
    return;
  }
  uint32_t current = leaf;
  // The child of the current node that does not contain the data
  ArrayDouble pred_new(n_classes());
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


//// This function actually compute the way the aggregate is computed along the path
//void TreeClassifier::get_aggregate_path(const SArrayDoublePtr features,
//                              SArrayDoublePtrList1D
//                              SArrayDouble2dPtr node_scores,
//                              SArrayDoublePtr aggregation_weights) {
//
//  ArrayDouble &x_t = *features;
//  uint32_t leaf = go_downwards(x_t, 0., true);
//  uint32_t current = leaf;
//  // The child of the current node that does not contain the data
//  ArrayDouble pred_new(n_classes());
//  ArrayDouble scores(n_classes());
//
//  aggregation_weights->fill(1.);
//
//  while (true) {
//    NodeClassifier &current_node = node(current);
//    if (current_node.is_leaf()) {
//      current_node.predict(scores);
//      std::cout << "node: " << current;
//      show_vector(scores);
//    } else {
//      float w = std::exp(current_node.weight() - current_node.weight_tree());
//      // Get the predictions of the current node
//      current_node.predict(pred_new);
//      std::cout << "current score: " << current;
//      show_vector(scores);
//      std::cout << "node scores: " << current;
//      show_vector(pred_new);
//      std::cout << "weight: " << 0.5 * w << std::endl;
//
//
//
//      for (uint8_t c = 0; c < n_classes(); ++c) {
//        scores[c] = 0.5 * w * pred_new[c] + (1 - 0.5 * w) * scores[c];
//
//
//      }
//      std::cout << "new score" << std::endl;
//      show_vector(scores);
//    }
//    // Root must be updated as well
//    if (current == 0) {
//      break;
//    }
//    current = current_node.parent();
//  }
//}


void TreeClassifier::reserve_nodes(uint32_t n_nodes) {
  nodes.reserve(n_nodes);
  // TODO: requires some thought...
  for (uint32_t i = 0; i < n_nodes; ++i) {
    nodes.emplace_back(*this, 0, 0);
  }
}

void TreeClassifier::create_root() {
  nodes.emplace_back(*this, 0, 0);
  _n_nodes++;
}

uint32_t TreeClassifier::add_node(uint32_t parent, float time) {
  if (_n_nodes < nodes.size()) {
    // We have enough nodes already, so let's use the last free one, and just update its time and parent
    node(_n_nodes).set_parent(parent).set_time(time);
    _n_nodes++;
    return _n_nodes - 1;
  } else {
    TICK_ERROR("OnlineForest: Something went wrong with nodes allocation !!!")
  }
}

inline uint32_t TreeClassifier::n_features() const {
  return _n_features;
}

inline uint8_t TreeClassifier::n_classes() const {
  return _n_classes;
}

inline uint32_t TreeClassifier::n_nodes() const {
  return _n_nodes;
}

inline uint32_t TreeClassifier::n_nodes_reserved() const {
  return static_cast<uint32_t>(nodes.size());
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

inline float TreeClassifier::step() const {
  return forest.step();
}

inline float TreeClassifier::dirichlet() const {
  return forest.dirichlet();
}

inline CriterionClassifier TreeClassifier::criterion() const {
  return forest.criterion();
}

inline bool TreeClassifier::use_aggregation() const {
  return forest.use_aggregation();
}

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
    SArrayFloatPtr nodes_weight,
    SArrayFloatPtr nodes_weight_tree,
    SArrayUShortPtr nodes_is_leaf,
    SArrayUInt2dPtr nodes_counts) {

  // std::cout << "void TreeClassifier::get_flat_nodes(" << std::endl;
  // std::cout << "n_nodes: " << _n_nodes << ", nodes.size(): " << nodes.size() << std::endl;

  for (uint32_t node_index = 0; node_index < _n_nodes; ++node_index) {
    // std::cout << "n_node= " << node_index << std::endl;
    NodeClassifier &node = nodes[node_index];
    (*nodes_parent)[node_index] = node.parent();
    (*nodes_left)[node_index] = node.left();
    (*nodes_right)[node_index] = node.right();
    (*nodes_feature)[node_index] = node.feature();
    (*nodes_threshold)[node_index] = node.threshold();
    (*nodes_time)[node_index] = node.time();
    (*nodes_depth)[node_index] = node.depth();
    // (*nodes_features_min)[n_node] = node.features_min();
    // (*nodes_features_max)[n_node] = node.features_max();
    (*nodes_n_samples)[node_index] = node.n_samples();
    (*nodes_weight)[node_index] = node.weight();
    (*nodes_weight_tree)[node_index] = node.weight_tree();
    // (*nodes_is_leaf)[n_node] = static_cast<ushort>(node.is_leaf());
    nodes_is_leaf->operator[](node_index) = static_cast<ushort>(node.is_leaf());
    (*nodes_is_leaf)[node_index] = static_cast<ushort>(node.is_leaf());
  }
}

/*********************************************************************************
 * OnlineForestClassifier methods
 *********************************************************************************/

// TODO: remove n_passes and subsampling
// TODO: Add the bootstrap option ? (supposedly useless)
// TODO: add a compute_depth option

OnlineForestClassifier::OnlineForestClassifier(uint32_t n_features,
                                               uint8_t n_classes,
                                               uint8_t n_trees,
                                               float step,
                                               CriterionClassifier criterion,
                                               FeatureImportanceType feature_importance_type,
                                               bool use_aggregation,
                                               float dirichlet,
                                               int32_t n_threads,
                                               int seed,
                                               bool verbose)
    : _n_features(n_features),
      _n_classes(n_classes),
      _n_trees(n_trees),
      _step(step),
      _criterion(criterion),
      _feature_importance_type(feature_importance_type),
      _use_aggregation(use_aggregation),
      _dirichlet(dirichlet),
      _n_threads(n_threads),
      _verbose(verbose),
      rand(seed) {
  // No iteration so far
  _iteration = 0;
  create_trees();
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

void OnlineForestClassifier::fit(const SArrayDouble2dPtr features,
                                 const SArrayDoublePtr labels) {
  uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
  uint32_t n_features = static_cast<uint32_t>(features->n_cols());
  if (_iteration == 0) {
    _n_features = n_features;
  } else {
    check_n_features(n_features, false);
  }
  for (TreeClassifier &tree : trees) {
    // Maximum number of nodes is now the current one + number of samples in this batch
    // TODO: IMPORTANT !!!! are we sure about this ?????
    // TODO: tree.reserve_nodes(tree.n_nodes() + 2 * n_samples - 1);
    // tree.reserve_nodes(2 * tree.n_nodes() + 2 * n_samples);
    if (tree.n_nodes() + 2 * n_samples > tree.n_nodes_reserved()) {
      tree.reserve_nodes(tree.n_nodes() + 2 * n_samples);
    }
    for (uint32_t i = 0; i < n_samples; ++i) {
      double label = (*labels)[i];
      check_label(label);
      tree.fit(view_row(*features, i), (*labels)[i]);
      _iteration++;
    }
  }
}

void OnlineForestClassifier::predict(const SArrayDouble2dPtr features,
                                     SArrayDouble2dPtr scores) {
  scores->fill(0.);
  uint32_t n_features = static_cast<uint32_t>(features->n_cols());
  check_n_features(n_features, true);
  if (_iteration > 0) {
    uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
    ArrayDouble scores_tree(_n_classes);
    scores_tree.fill(0.);
    ArrayDouble scores_forest(_n_classes);
    scores_forest.fill(0.);
    for (uint32_t i = 0; i < n_samples; ++i) {
      // The prediction is simply the average of the predictions
      ArrayDouble scores_i = view_row(*scores, i);
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
  _iteration = 0;
}

void OnlineForestClassifier::print() {
  for (TreeClassifier &tree: trees) {
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

bool OnlineForestClassifier::verbose() const {
  return _verbose;
}

OnlineForestClassifier &OnlineForestClassifier::set_verbose(bool verbose) {
  _verbose = verbose;
  return *this;
}

bool OnlineForestClassifier::use_aggregation() const {
  return _use_aggregation;
}

float OnlineForestClassifier::step() const {
  return _step;
}

OnlineForestClassifier &OnlineForestClassifier::set_step(const float step) {
  _step = step;
  return *this;
}

uint32_t OnlineForestClassifier::n_samples() const {
  if (_iteration > 0) {
    return _iteration;
  } else {
    TICK_ERROR("You must call ``fit`` before asking for ``n_samples``.")
  }
}

uint32_t OnlineForestClassifier::n_features() const {
  return _n_features;
}

void OnlineForestClassifier::check_n_features(uint32_t n_features, bool predict) const {
  if (n_features != _n_features) {
    if (predict) {
      TICK_ERROR("Wrong number of features: trained with " + std::to_string(_n_features)
                     + " features, but received " + std::to_string(n_features) + " features for prediction");
    } else {
      TICK_ERROR("Wrong number of features: started to train with " + std::to_string(_n_features)
                     + " features, but received " + std::to_string(n_features) + " afterwards");
    }
  }
}

void OnlineForestClassifier::check_label(double label) const {
  double iptr;
  double fptr = std::modf(label, &iptr);
  if (fptr != 0) {
    TICK_ERROR("Wrong label type: received " + std::to_string(label) + " for a classification problem");
  }
  if ((label < 0) || (label >= _n_classes)) {
    TICK_ERROR("Wrong label value: received " + std::to_string(label) + " while training for classification with "
                   + std::to_string(_n_classes) + " classes.");
  }
}

uint8_t OnlineForestClassifier::n_classes() const {
  return _n_classes;
}

uint8_t OnlineForestClassifier::n_trees() const {
  return _n_trees;
}

double OnlineForestClassifier::given_feature_importances(const ulong j) const {
  return static_cast<double>(_given_feature_importances[j]);
}

int32_t OnlineForestClassifier::n_threads() const {
  return _n_threads;
}

CriterionClassifier OnlineForestClassifier::criterion() const {
  return _criterion;
}

int OnlineForestClassifier::seed() const {
  return _seed;
}

OnlineForestClassifier &OnlineForestClassifier::set_seed(int seed) {
  _seed = seed;
  rand.reseed(seed);
  return *this;
}

OnlineForestClassifier &OnlineForestClassifier::set_n_threads(int32_t n_threads) {
  _n_threads = n_threads;
  return *this;
}

OnlineForestClassifier &OnlineForestClassifier::set_criterion(CriterionClassifier criterion) {
  _criterion = criterion;
  return *this;
}

FeatureImportanceType OnlineForestClassifier::feature_importance_type() const {
  return _feature_importance_type;
}

OnlineForestClassifier &OnlineForestClassifier::set_given_feature_importances(const ArrayDouble &feature_importances) {
  for (ulong j = 0; j < n_features(); ++j) {
    _given_feature_importances[j] = static_cast<float>(feature_importances[j]);
  }
  return *this;
}

float OnlineForestClassifier::dirichlet() const {
  return _dirichlet;
}

OnlineForestClassifier &OnlineForestClassifier::set_dirichlet(const float dirichlet) {
  // TODO: check that it's > 0
  _dirichlet = dirichlet;
  return *this;
}

void OnlineForestClassifier::get_feature_importances(SArrayDoublePtr feature_importances) {
  if (_feature_importance_type == FeatureImportanceType::estimated) {
    const float a = static_cast<float>(1) / n_trees();
    ArrayFloat importances(_n_features);
    importances.fill(0);
    for (TreeClassifier &tree : trees) {
      importances.mult_incr(tree.feature_importances(), a);
    }
    importances /= (importances.sum());
    for (ulong j = 0; j < _n_features; ++j) {
      (*feature_importances)[j] = static_cast<double>(importances[j]);
    }
  } else {
    if (_feature_importance_type == FeatureImportanceType::given) {
      for (ulong j = 0; j < _n_features; ++j) {
        (*feature_importances)[j] = static_cast<double>(_given_feature_importances[j]);
      }
    }
  }
}

uint32_t OnlineForestClassifier::get_path_depth(const uint8_t tree, const SArrayDoublePtr x_t) {
  return trees[tree].get_path_depth(*x_t);
}

// Get the path of x_t
void OnlineForestClassifier::get_path(const uint8_t tree, const SArrayDoublePtr x_t, SArrayUIntPtr path) {
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
    SArrayFloatPtr nodes_weight,
    SArrayFloatPtr nodes_weight_tree,
    SArrayUShortPtr nodes_is_leaf,
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
      nodes_weight,
      nodes_weight_tree,
      nodes_is_leaf,
      nodes_counts);
}
