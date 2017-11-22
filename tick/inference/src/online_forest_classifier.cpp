
// License: BSD 3 clause

#include "online_forest_classifier.h"

/*********************************************************************************
 * NodeClassifier methods
 *********************************************************************************/

NodeClassifier::NodeClassifier(TreeClassifier &tree, uint32_t parent, double time)
    : _tree(tree) {
  _parent = parent;
  _time = time;
  _left = 0;
  _right = 0;
  _n_samples = 0;
  _is_leaf = true;
  _weight = 0;
  _weight_tree = 0;
  _counts = ArrayULong(n_classes());
  _counts.fill(0);
}

NodeClassifier::NodeClassifier(const NodeClassifier &node)
    : _tree(node._tree),
      _parent(node._parent), _left(node._left), _right(node._right),
      _feature(node._feature), _threshold(node._threshold),
      _time(node._time), _features_min(node._features_min), _features_max(node._features_max),
      _n_samples(node._n_samples),
      _x_t(node._x_t),
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
  _features_min = node._features_min;
  _features_max = node._features_max;
  _n_samples = node._n_samples;
  _x_t = node._x_t;
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
  _features_min = node._features_min;
  _features_max = node._features_max;
  _n_samples = node._n_samples;
  _x_t = node._x_t;
  _y_t = node._y_t;
  _weight = node._weight;
  _weight_tree = node._weight_tree;
  _is_leaf = node._is_leaf;
  _counts = node._counts;
  return *this;
}

void NodeClassifier::update_downwards(const ArrayDouble &x_t, const double y_t) {
  _n_samples++;
  _weight -= step() * loss(y_t);
  update_predict(y_t);
}

bool NodeClassifier::is_same(const ArrayDouble &x_t) {
  if (_is_leaf) {
    for (uint32_t j = 0; j < n_features(); ++j) {
      double delta = std::abs(x_t[j] - _x_t[j]);
      if (delta > 0.) {
        return false;
      }
    }
    return true;
  } else {
    TICK_ERROR("NodeClassifier::is_same: node is not a leaf !")
  }
}

void NodeClassifier::update_upwards() {
  if (_is_leaf) {
    _weight_tree = _weight;
  } else {
    _weight_tree = log_sum_2_exp(_weight, node(_left).weight_tree() + node(_right).weight_tree());
  }
}

void NodeClassifier::update_predict(const double y_t) {
  // We update the counts for the class y_t
  _counts[static_cast<uint8_t>(y_t)]++;
}

void NodeClassifier::update_range(const ArrayDouble &x_t) {
  if (_n_samples == 0) {
    _features_min = x_t;
    _features_max = x_t;
  } else {
    for(uint32_t j = 0; j < n_features(); ++j) {
      double x_tj = x_t[j];
      if (x_tj < _features_min[j]) {
        _features_min[j] = x_tj;
      }
      if (x_tj > _features_max[j]) {
        _features_max[j] = x_tj;
      }
    }
  }
}

double NodeClassifier::score(uint8_t c) const {
  // Using Dirichet(1/2, ... 1/2) prior
  return static_cast<double>(2 * _counts[c] + 1) / (2 * _n_samples + n_classes());
}

inline void NodeClassifier::predict(ArrayDouble& scores) const {
  for (uint8_t c=0; c < n_classes(); ++c) {
    scores[c] = score(c);
  }
}

double NodeClassifier::loss(const double y_t) {
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

uint8_t  NodeClassifier::n_classes() const {
  return _tree.n_classes();
}

inline double NodeClassifier::step() const {
  return _tree.step();
}

inline uint32_t NodeClassifier::parent() const {
  return _parent;
}

inline NodeClassifier& NodeClassifier::set_parent(uint32_t parent) {
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

inline double NodeClassifier::threshold() const {
  return _threshold;
}

inline NodeClassifier &NodeClassifier::set_threshold(double threshold) {
  _threshold = threshold;
  return *this;
}

inline double NodeClassifier::time() const {
  return _time;
}

inline NodeClassifier &NodeClassifier::set_time(double time) {
  _time = time;
  return *this;
}

inline double NodeClassifier::features_min(const uint32_t j) const {
  return _features_min[j];
}

inline NodeClassifier & NodeClassifier::set_features_min(const ArrayDouble &features_min) {
  _features_min = features_min;
  return *this;
}

inline double NodeClassifier::features_max(const uint32_t j) const {
  return _features_max[j];
}

inline NodeClassifier & NodeClassifier::set_features_max(const ArrayDouble &features_max) {
  _features_max = features_max;
  return *this;
}

inline uint32_t NodeClassifier::n_samples() const {
  return _n_samples;
}

inline NodeClassifier &NodeClassifier::set_n_samples(uint32_t n_samples) {
  _n_samples = n_samples;
  return *this;
}

inline double NodeClassifier::weight() const {
  return _weight;
}

inline NodeClassifier &NodeClassifier::set_weight(double weight) {
  _weight = weight;
  return *this;
}

inline double NodeClassifier::weight_tree() const {
  return _weight_tree;
}

inline NodeClassifier &NodeClassifier::set_weight_tree(double weight_tree) {
  _weight_tree = weight_tree;
  return *this;
}

inline const ArrayDouble &NodeClassifier::x_t() const {
  return _x_t;
}

inline NodeClassifier &NodeClassifier::set_x_t(const ArrayDouble &x_t) {
  _x_t = x_t;
  return *this;
}

inline double NodeClassifier::y_t() const {
  return _y_t;
}

inline NodeClassifier &NodeClassifier::set_y_t(const double y_t) {
  _y_t = y_t;
  return *this;
}

void NodeClassifier::print() {
  std::cout << "Node(parent: " << _parent
      << ", left: " << _left
      << ", right: " << _right
      << ", time: " << std::setprecision(2) << _time
      << ", n_samples: " << _n_samples
      << ", is_leaf: " << _is_leaf
      << ", feature: " << _feature
      << ", thresh: " << _threshold
      << ", scores: [" << std::setprecision(2) << score(0) << ", " << std::setprecision(2) << score(1) << "]"
      << ", counts: [" << std::setprecision(2) << _counts[0] << ", " << std::setprecision(2) << _counts[1] << "]";
      if (_n_samples > 0) {
        std::cout << ", min: [" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2) << _features_min[1] << "]"
                  << ", max: [" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2) << _features_max[1] << "]";

      }
      std::cout << ", weight: " << _weight
      << ", weight_tree: " << _weight_tree
      << ")\n";
}

/*********************************************************************************
* TreeClassifier methods
*********************************************************************************/

TreeClassifier::TreeClassifier(const TreeClassifier &tree)
    : forest(tree.forest), nodes(tree.nodes) {}

TreeClassifier::TreeClassifier(const TreeClassifier &&tree)
    : forest(tree.forest), nodes(tree.nodes) {}

TreeClassifier::TreeClassifier(OnlineForestClassifier &forest) : forest(forest) {
  // TODO: pre-allocate the vector to make things faster ?
  add_node(0);
}

//uint32_t TreeClassifier::split_leaf(uint32_t index, const ArrayDouble &x_t, double y_t) {
//  // std::cout << "Splitting node " << index << std::endl;
//  uint32_t left = add_node(index);
//  uint32_t right = add_node(index);
//  node(index).set_left(left).set_right(right).set_is_leaf(false);
//
//  // std::cout << "n_features(): " << n_features() << std::endl;
//  ArrayDouble diff(n_features());
//  for(uint32_t j = 0; j < n_features(); ++j) {
//    // std::cout << "j: " << j;
//    diff[j] = std::abs(node(index).x_t()[j] - x_t[j]);
//  }
//  // std::cout << std::endl;
//  diff /= diff.sum();
//  // diff.print();
//  // std::cout << "diff.sum=" << diff.sum() << std::endl;
//
//  // TODO: better feature sampling
//  // ulong feature = forest.sample_feature_bis();
//  // ulong feature = forest.sample_feature();
//
//  uint32_t feature = forest.sample_feature(diff);
//
//  // std::cout << "feature: " << feature << std::endl;
//
//  double x1_tj = x_t[feature];
//  double x2_tj = node(index).x_t()[feature];
//  double threshold;
//
//  // The leaf that contains the passed sample (x_t, y_t)
//  uint32_t data_leaf;
//  uint32_t other_leaf;
//
//  // std::cout << "x1_tj= " << x1_tj << " x2_tj= " << x2_tj << " threshold= " << threshold << std::endl;
//  // TODO: what if x1_tj == x2_tj. Must be taken care of by sample_feature()
//  if (x1_tj < x2_tj) {
//    threshold = forest.sample_threshold(x1_tj, x2_tj);
//    data_leaf = left;
//    other_leaf = right;
//  } else {
//    threshold = forest.sample_threshold(x2_tj, x1_tj);
//    data_leaf = right;
//    other_leaf = left;
//  }
//  // TODO: code a move_sample
//  NodeClassifier & current_node = node(index);
//  NodeClassifier & data_node = node(data_leaf);
//  NodeClassifier & other_node = node(other_leaf);
//  current_node.set_feature(feature).set_threshold(threshold);
//  // We pass the sample to the new leaves, and initialize the _label_average with the value
//  data_node.set_x_t(x_t).set_y_t(y_t);
//
//  // other_node.set_x_t(current_node.x_t()).set_y_t(current_node.y_t());
//  other_node.set_x_t(current_node.x_t()).set_y_t(current_node.y_t());
//
//  // Update downwards of v'
//  other_node.update_downwards(current_node.x_t(), current_node.y_t());
//  // Update upwards of v': it's a leaf
//  other_node.update_upwards();
//  // node(other_leaf).set_weight_tree(node(other_leaf).weight());
//  // Update downwards of v''
//  data_node.update_downwards(x_t, y_t);
//  // Note: the update_up of v'' is done in the go_up method, called in fit()
//  // std::cout << "Done splitting node." << std::endl;
//  return data_leaf;
//}

void TreeClassifier::extend_range(uint32_t node_index, const ArrayDouble &x_t, const double y_t) {
  // std::cout << "Extending the range of: " << index << std::endl;
  NodeClassifier &current_node = node(node_index);
  if(current_node.n_samples() == 0) {
    // The node is a leaf with no sample point, so it does not have a range
    // In this case we just initialize the range with the given feature
    // This node will then be updated by the call to update_downwards in go_downwards
    current_node.set_features_min(x_t);
    current_node.set_features_max(x_t);
  } else {
    // std::cout << "Computing extension" << std::endl;
    ArrayDouble extension(n_features());
    double extensions_sum = 0;
    for(uint32_t j =0; j < n_features(); ++j) {
      double x_tj = x_t[j];
      double feature_min_j = current_node.features_min(j);
      double feature_max_j = current_node.features_max(j);
      if(x_tj < feature_min_j) {
        extension[j] = feature_min_j - x_tj;
        extensions_sum += feature_min_j - x_tj;
      } else {
        if (x_tj > feature_max_j) {
          extension[j] = x_tj - feature_max_j;
          extensions_sum += x_tj - feature_max_j;
        } else {
          extension[j] = 0;
        }
      }
    }
//    std::cout << "extension: [" << extension[0] << ", " << std::setprecision(2) << extension[1] << "]" << std::endl;
//    std::cout << "extension_sum: " << std::setprecision(2) << extensions_sum << std::endl;
//    std::cout << "... Done computing extension." << std::endl;

    // If the sample x_t extends the current range of the node
    if(extensions_sum > 0) {
      // std::cout << "Extension non-zero, considering the possibility of a split" << std::endl;
      bool do_split;
      double time = current_node.time();
      double T = forest.sample_exponential(extensions_sum);
      // std::cout << "time: " << std::setprecision(2) << time << ", T: " << std::setprecision(2) << T << std::endl;
      // Let us determine if we need to split the node or not
      if (current_node.is_leaf()) {
        // std::cout << "I'll split the node since it's a leaf" << std::endl;
        do_split = true;
      } else {
        // Same as node(current_node.right()).time();
        double child_time = node(current_node.left()).time();
        // Sample a exponential random variable with intensity
        if (time + T < child_time) {
          // std::cout << "  I'll split since time + T < child_time with child_time: " << child_time << std::endl;
          do_split = true;
        } else {
          // std::cout << "I won't split since time + T >= child_time with child_time: " << child_time << std::endl;
          do_split = false;
        }
      }
      if (do_split) {
        // std::cout << "Starting the splitting of node: " << node_index << std::endl;
        // Sample the splitting feature with a probability proportional to the range extensions
        ArrayDouble probabilities = extension;
        probabilities /= extensions_sum;
        // std::cout << "using the probabilities: [" << std::setprecision(2) << probabilities[0] << ", " << std::setprecision(2) << probabilities[1] << "]" << std::endl;
        uint32_t feature = forest.sample_feature(probabilities);
        // std::cout << "sampled feature: " << feature << std::endl;
        double threshold;
        // Is the extension on the right side ?
        bool is_right_extension = x_t[feature] > current_node.features_max(feature);
        // Create new nodes
        uint32_t left_new = add_node(node_index, time + T);
        uint32_t right_new = add_node(node_index, time + T);
        if(is_right_extension) {
          // std::cout << "extension is on the right" << std::endl;
          threshold = forest.sample_threshold(node(node_index).features_max(feature), x_t[feature]);
          // std::cout << "sample inside the extension the threshold: " << threshold << std::endl;
          // left_new is the same as node_index, excepted for the parent, time and the fact that it's not a leaf
          // std::cout << "Let's copy the current node in the left child" << threshold << std::endl;
          node(left_new) = node(node_index);
          // donc faut remettre le bon parent et le bon temps
          // TODO: set_is_leaf useless for left_new since it's a copy of node_index
          // std::cout << "Let's the update the left child" << std::endl;
          node(left_new).set_parent(node_index).set_time(time + T);
          // right_new doit avoir comme parent node_index
          // std::cout << "Let's the update the right child" << std::endl;
          node(right_new).set_parent(node_index).set_time(time + T);
          // We must tell the old childs that they have a new parent, if the current node is not a leaf
          if(!node(node_index).is_leaf()) {
            // std::cout << "The current node is not a leaf, so let's not forget to update the old childs" << std::endl;
            node(node(node_index).left()).set_parent(left_new);
            node(node(node_index).right()).set_parent(left_new);
          }
          // TODO: faut retourner right_new dans ce cas ?
        } else {
          // std::cout << "extension is on the left" << std::endl;
          threshold = forest.sample_threshold(x_t[feature], node(node_index).features_min(feature));
          node(right_new) = node(node_index);
          node(right_new).set_parent(node_index).set_time(time + T);
          node(left_new).set_parent(node_index).set_time(time + T);
          if(!node(node_index).is_leaf()) {
            node(node(node_index).left()).set_parent(right_new);
            node(node(node_index).right()).set_parent(right_new);
          }
        }
        // We update the splitting feature, threshold, and childs of the current index
        node(node_index).set_feature(feature).set_threshold(threshold).set_left(left_new)
            .set_right(right_new).set_is_leaf(false);
      }
      // Update the range of the node here
      node(node_index).update_range(x_t);
    }
  }
  // std::cout << "...Done extending the range." << std::endl;
}


uint32_t TreeClassifier::go_downwards(const ArrayDouble &x_t, double y_t, bool predict) {
  // Find the leaf that contains the sample
  // Start at the root. Index of the root is always 0
  // If predict == true, this call to find_leaf is for
  // prediction only, so that no leaf update and splits can be done
  // std::cout << "Going downwards" << std::endl;
  uint32_t index_current_node = 0;
  bool is_leaf = false;
  while (!is_leaf) {
    // Get the current node
    // NodeClassifier &current_node = node(index_current_node);
    if (!predict) {
      // Extend the range and eventually split the current node
      extend_range(index_current_node, x_t, y_t);
      // Update the current node
      node(index_current_node).update_downwards(x_t, y_t);
    }
    // Is the node a leaf ?
    NodeClassifier &current_node = node(index_current_node);
    is_leaf = current_node.is_leaf();
    if (!is_leaf) {
      if (x_t[current_node.feature()] <= current_node.threshold()) {
        index_current_node = current_node.left();
      } else {
        index_current_node = current_node.right();
      }
    }
  }
  // std::cout << "...Done going downwards." << std::endl;
  return index_current_node;
}

void TreeClassifier::go_upwards(uint32_t leaf_index) {
  // std::cout << "Going upwards" << std::endl;
  uint32_t current = leaf_index;
  while (true) {
    NodeClassifier &current_node = node(current);
    current_node.update_upwards();
    if (current == 0) {
      // std::cout << "...Done going upwards." << std::endl;
      break;
    }
    // We must update the root node
    current = node(current).parent();
  }
}

inline uint32_t TreeClassifier::n_nodes() const {
  return _n_nodes;
}

uint32_t TreeClassifier::n_leaves() const {
  uint32_t n_leaves = 0;
  for(const NodeClassifier &node: nodes) {
    if(node.is_leaf()) {
      ++n_leaves;
    }
  }
  return n_leaves;
}

void TreeClassifier::fit(const ArrayDouble &x_t, double y_t) {
  // TODO: Test that the size does not change within successive calls to fit
  // std::cout << "------------------------------------------" << std::endl;
  // std::cout << "iteration: " << iteration << std::endl;
  // std::cout << "x_t: [" << std::setprecision(2) << x_t[0] << ", " << std::setprecision(2) << x_t[1] << "]" << std::endl;
  // print();
  uint32_t leaf = go_downwards(x_t, y_t, false);
  go_upwards(leaf);
  iteration++;
}

void TreeClassifier::predict(const ArrayDouble &x_t, ArrayDouble& scores, bool use_aggregation) {
  // std::cout << "Going downwards" << std::endl;
  uint32_t leaf = go_downwards(x_t, 0., true);

  if(!use_aggregation) {
    node(leaf).predict(scores);
    return;
  }

  // std::cout << "Done." << std::endl;
  uint32_t current = leaf;
  // The child of the current node that does not contain the data
  ArrayDouble pred_new(n_classes());
  while (true) {
    // std::cout << "node: " << current << std::endl;
    NodeClassifier &current_node = node(current);
    if (current_node.is_leaf()) {
      current_node.predict(scores);
    } else {
      double w = std::exp(current_node.weight() - current_node.weight_tree());
      // Get the predictions of the current node
      current_node.predict(pred_new);
      for(uint8_t c = 0; c < n_classes(); ++c) {
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

uint32_t TreeClassifier::add_node(uint32_t parent, double time) {
  // std::cout << "Adding node with parent " << parent << std::endl;
  nodes.emplace_back(*this, parent, time);
  // std::cout << "Done." << std::endl;
  return _n_nodes++;
}

inline uint32_t TreeClassifier::n_features() const {
  return forest.n_features();
}

inline uint8_t TreeClassifier::n_classes() const {
  return forest.n_classes();
}

inline double TreeClassifier::step() const {
  return forest.step();
}

inline CriterionClassifier TreeClassifier::criterion() const {
  return forest.criterion();
}

/*********************************************************************************
 * OnlineForestClassifier methods
 *********************************************************************************/

OnlineForestClassifier::OnlineForestClassifier(uint32_t n_trees,
                                               uint8_t n_classes,
                                               double step,
                                               CriterionClassifier criterion,
                                               bool use_aggregation,
                                               int32_t n_threads,
                                               int seed,
                                               bool verbose)
    : _n_trees(n_trees), _n_classes(n_classes), _n_threads(n_threads),
      _criterion(criterion), _use_aggregation(use_aggregation), _step(step), _verbose(verbose), trees() {
  // No iteration so far
  _iteration = 0;

//  std::cout << "sizeof(float): " << sizeof(float) << std::endl;
//  std::cout << "sizeof(double): " << sizeof(double) << std::endl;
//  std::cout << "sizeof(uint8_t): " << sizeof(uint8_t) << std::endl;
//  std::cout << "sizeof(uint16_t): " << sizeof(uint16_t) << std::endl;
//  std::cout << "sizeof(uint32_t): " << sizeof(uint32_t) << std::endl;
//  std::cout << "sizeof(long): " << sizeof(long) << std::endl;
//  std::cout << "sizeof(ulong): " << sizeof(ulong) << std::endl;

  create_trees();
  // Seed the random number generators
  set_seed(seed);
}

OnlineForestClassifier::~OnlineForestClassifier() {}

void OnlineForestClassifier::create_trees() {
  // Just in case...
  trees.clear();
  trees.reserve(_n_trees);
  for (uint32_t i = 0; i < _n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

void OnlineForestClassifier::fit(const SArrayDouble2dPtr features,
                                const SArrayDoublePtr labels) {
  // std::cout << "OnlineForestClassifier::fit" << std::endl;

  _features = features;
  _labels = labels;

  uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
  uint32_t n_features = static_cast<uint32_t>(features->n_cols());
  set_n_features(n_features);
  for (uint32_t i = 0; i < n_samples; ++i) {
    for (TreeClassifier &tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(view_row(*features, i), (*labels)[i]);
    }
    _iteration++;
  }
  // std::cout << "Done OnlineForestClassifier::fit" << std::endl;
}

void OnlineForestClassifier::predict(const SArrayDouble2dPtr features,
                                     SArrayDouble2dPtr predictions) {
  predictions->fill(0.);
  if (_iteration > 0) {
    uint32_t n_samples = static_cast<uint32_t>(features->n_rows());
    ArrayDouble scores_tree(_n_classes);
    scores_tree.fill(0.);
    ArrayDouble scores_forest(_n_classes);
    scores_forest.fill(0.);
    for (uint32_t i = 0; i < n_samples; ++i) {
      // The prediction is simply the average of the predictions
      ArrayDouble scores_i = view_row(*predictions, i);
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

inline uint32_t OnlineForestClassifier::sample_feature() {
  return rand.uniform_int(static_cast<uint32_t>(0), n_features() - 1);
}

inline uint32_t OnlineForestClassifier::sample_feature_bis() {
  return rand.discrete(_feature_importances);
}

inline double OnlineForestClassifier::sample_exponential(double intensity) {
  return rand.exponential(intensity);
}

inline uint32_t OnlineForestClassifier::sample_feature(const ArrayDouble & prob) {
  ArrayDouble my_prob = prob;
  for(uint32_t j = 0; j < n_features(); ++j) {
    // my_prob[j] *= _feature_importances[j];
    my_prob[j] = _feature_importances[j];
  }
  my_prob /= my_prob.sum();
  return rand.discrete(my_prob);
}

inline double OnlineForestClassifier::sample_threshold(double left, double right) {
  return rand.uniform(left, right);
}

//inline bool OnlineForestClassifier::verbose() const {
//  return _verbose;
//}
//
//inline OnlineForestClassifier &OnlineForestClassifier::set_verbose(bool verbose) {
//  _verbose = verbose;
//  return *this;
//}
