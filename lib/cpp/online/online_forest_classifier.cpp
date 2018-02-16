// License: BSD 3 clause

#include "tick/online/online_forest_classifier.h"

/*********************************************************************************
 * NodeClassifier methods
 *********************************************************************************/

NodeClassifier::NodeClassifier(TreeClassifier &tree, uint32_t parent, float time)
    : _tree(tree), _parent(parent), _left(0), _right(0), _time(time),
      _features_min(tree.n_features()), _features_max(tree.n_features()),
      _n_samples(0), _is_leaf(true), _weight(0), _weight_tree(0),
      _counts(tree.n_classes()) {
  _counts.fill(0);
}

NodeClassifier::NodeClassifier(const NodeClassifier &node)
    : _tree(node._tree), _parent(node._parent), _left(node._left), _right(node._right),
      _feature(node._feature), _threshold(node._threshold),
      _time(node._time), _features_min(node._features_min), _features_max(node._features_max),
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

float NodeClassifier::update_downwards(const ArrayDouble &x_t, const double y_t) {
  _n_samples++;
  float loss_t = loss(y_t);
  if (use_aggregation()) {
    _weight -= step() * loss_t;
  }
  update_predict(y_t);
  // We return the loss before updating the predictor of the node in order to
  // update the feature importance in TreeClassifier::go_downwards
  return loss_t;
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

float NodeClassifier::score(uint8_t c) const {
  // Using the Dirichet prior
  return (_counts[c] + dirichlet()) / (_n_samples + dirichlet() * n_classes());
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
            << ", feature: " << _feature
            << ", thresh: " << _threshold
            << ", scores: [" << std::setprecision(2) << score(0) << ", " << std::setprecision(2) << score(1) << "]"
            << ", counts: [" << std::setprecision(2) << _counts[0] << ", " << std::setprecision(2) << _counts[1] << "]";
  if (_n_samples > 0) {
    std::cout << ", min: [" << std::setprecision(2) << _features_min[0] << ", " << std::setprecision(2)
              << _features_min[1] << "]"
              << ", max: [" << std::setprecision(2) << _features_max[0] << ", " << std::setprecision(2)
              << _features_max[1] << "]";
  }
  std::cout << ", weight: " << _weight
            << ", weight_tree: " << _weight_tree
            << ")\n";
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

void TreeClassifier::extend_range(uint32_t node_index, const ArrayDouble &x_t, const double y_t) {
  NodeClassifier &current_node = node(node_index);
  if (current_node.n_samples() == 0) {
    // The node is a leaf with no sample point, so it does not have a range
    // In this case we just initialize the range with the given feature.
    // This node will then be updated by the call to update_downwards in go_downwards
    current_node.update_range(x_t);
  } else {
    // A vector that will hold the intensities of each feature.
    // The intensity of a feature is measured by the product
    // between the square root of the feature importance and the range extension at this node...
    float intensities_sum = 0;
    for (uint32_t j = 0; j < _n_features; ++j) {
      float x_tj = static_cast<float>(x_t[j]);
      float feature_min_j = current_node.features_min(j);
      float feature_max_j = current_node.features_max(j);
      // TODO: several choices are available here...
      // Intensity will be used to compute the probabilities used for
      // sampling the feature. It contains the extension of the range wrt each feature
      // TODO: ouch !!! Some extra work on feature selection is welcome
      // float intensity = feature_importance(j) / (iteration + 1);
      float intensity = 1;
      if (x_tj < feature_min_j) {
        float diff = intensity * (feature_min_j - x_tj);
        intensities[j] = diff;
        intensities_sum += diff;
      } else {
        if (x_tj > feature_max_j) {
          float diff = intensity * (x_tj - feature_max_j);
          intensities[j] = diff;
          intensities_sum += diff;
        } else {
          intensities[j] = 0;
        }
      }
    }
    // If the sample x_t extends the current range of the node
    if (intensities_sum > 0) {
      bool do_split;
      float time, T;
      // Let us determine if we need to split the node or not
      if (current_node.is_leaf()) {
        // If the node is a leaf we must split it
        do_split = true;
      } else {
        time = current_node.time();
        // NB: time = current_node.time() gives the same thing...
        T = forest.sample_exponential(intensities_sum);
        float child_time = node(current_node.left()).time();
        // Sample a exponential random variable with intensity
        if (time + T < child_time) {
          // Another Mondrian dark magic :) This is the rejection rule used
          // in the Mondrian process
          do_split = true;
        } else {
          do_split = false;
        }
      }
      if (do_split) {
        // Sample the splitting feature with a probability proportional to the range extensions
        intensities /= intensities.sum();
        uint32_t feature = forest.sample_feature(intensities);
        float threshold;
        // Is the extension on the right side ?
        bool is_right_extension = x_t[feature] > current_node.features_max(feature);
        // Create new nodes
        uint32_t left_new = add_node(node_index, time + T);
        uint32_t right_new = add_node(node_index, time + T);
        // Let's take again the current node (adding node might lead to re-allocations in the nodes std::vector)
        current_node = node(node_index);
        NodeClassifier &left_new_node = node(left_new);
        NodeClassifier &right_new_node = node(right_new);
        // The value of the feature
        float x_tf = static_cast<float>(x_t[feature]);
        if (is_right_extension) {
          // Sample a threshold uniformly in the range extension: the magic of the Mondrian process
          threshold = forest.sample_threshold(current_node.features_max(feature), x_tf);
          // left_new is the same as node_index, excepted for the parent, time and the fact that it's not a leaf
          left_new_node = current_node;
          // so we need to put back the correct parent and time
          left_new_node.set_parent(node_index).set_time(time + T);
          // right_new doit avoir comme parent node_index
          right_new_node.set_parent(node_index).set_time(time + T);
          // We must tell the old childs that they have a new parent, if the current node is not a leaf
          if (!current_node.is_leaf()) {
            node(current_node.left()).set_parent(left_new);
            node(current_node.right()).set_parent(left_new);
          }
        } else {
          threshold = forest.sample_threshold(x_t[feature], current_node.features_min(feature));
          right_new_node = current_node;
          right_new_node.set_parent(node_index).set_time(time + T);
          left_new_node.set_parent(node_index).set_time(time + T);
          if (!current_node.is_leaf()) {
            node(current_node.left()).set_parent(right_new);
            node(current_node.right()).set_parent(right_new);
          }
        }
        // We update the splitting feature, threshold, and childs of the current index
        current_node.set_feature(feature).set_threshold(threshold).set_left(left_new)
            .set_right(right_new).set_is_leaf(false);
      }
      // Update the range of the node here
      current_node.update_range(x_t);
    }
  }
}

uint32_t TreeClassifier::go_downwards(const ArrayDouble &x_t, double y_t, bool predict) {
  // Find the leaf that contains the sample. Start at the root. Index of the root is always 0.
  // If predict == true, this is for prediction only, so no leaf update and splits can be done.
  uint32_t index_current_node = 0;
  bool is_leaf = false;
  float loss_t = 0;
  uint32_t feature = 0;
  while (!is_leaf) {
    if (!predict) {
      // Extend the range and eventually split the current node
      extend_range(index_current_node, x_t, y_t);
      // Update the current node. We get the loss for this point before the node update
      // to compute feature importance below
      NodeClassifier &current_node = node(index_current_node);
      feature = current_node.feature();
      loss_t = current_node.update_downwards(x_t, y_t);
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
      if (!predict) {
        // Compute the difference with the loss of the child
        loss_t -= node(index_current_node).loss(y_t);
        if (loss_t > 0) {
          feature_importances_[feature] += loss_t;
        }
      }
    }
  }
  return index_current_node;
}

void TreeClassifier::go_upwards(uint32_t leaf_index) {
  uint32_t current = leaf_index;
  while (true) {
    NodeClassifier &current_node = node(current);
    current_node.update_upwards();
    if (current == 0) {
      break;
    }
    // We must update the root node
    current = node(current).parent();
  }
}

void TreeClassifier::print() {
  std::cout << "Tree(n_nodes: " << _n_nodes << std::endl;
  std::cout << " ";
  uint32_t index = 0;
  for (NodeClassifier &node : nodes) {
    std::cout << "index: " << index << " ";
    node.print();
    index++;
  }
  std::cout << ")" << std::endl;
}

void TreeClassifier::fit(const ArrayDouble &x_t, double y_t) {
  uint32_t leaf = go_downwards(x_t, y_t, false);
  if (use_aggregation()) {
    go_upwards(leaf);
  }
  iteration++;
}

void TreeClassifier::predict(const ArrayDouble &x_t, ArrayDouble &scores, bool use_aggregation) {
  uint32_t leaf = go_downwards(x_t, 0., true);
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

void TreeClassifier::reserve_nodes(uint32_t n_nodes) {
  nodes.reserve(n_nodes);
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
    TICK_ERROR('OnlineForest: Something went wrong with nodes allocation !!!')
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

uint32_t TreeClassifier::n_leaves() const {
  uint32_t n_leaves = 0;
  for (const NodeClassifier &node: nodes) {
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

/*********************************************************************************
 * OnlineForestClassifier methods
 *********************************************************************************/

// TODO: remove n_passes and subsampling
// Add the bootstrap option ?

OnlineForestClassifier::OnlineForestClassifier(uint32_t n_features,
                                               uint8_t n_classes,
                                               uint8_t n_trees,
                                               uint8_t n_passes,
                                               float step,
                                               CriterionClassifier criterion,
                                               FeatureImportanceType feature_importance_type,
                                               bool use_aggregation,
                                               double subsampling,
                                               float dirichlet,
                                               int32_t n_threads,
                                               int seed,
                                               bool verbose)
    : _n_features(n_features),
      _n_classes(n_classes),
      _n_trees(n_trees),
      _n_passes(n_passes),
      _step(step),
      _criterion(criterion),
      _feature_importance_type(feature_importance_type),
      _use_aggregation(use_aggregation),
      _subsampling(subsampling),
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
    tree.reserve_nodes(2 * tree.n_nodes() + 2 * n_samples);
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
