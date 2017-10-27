
// License: BSD 3 clause

#include "online_forest.h"

Node::Node(Tree &tree, ulong index, ulong parent, ulong creation_time)
    : in_tree(tree) {
  _n_samples = 0;
  _is_leaf = true;
  _left = 0;
  _right = 0;
  this->_index = index;
  this->_parent = parent;
  this->creation_time = creation_time;
  _features_max = ArrayDouble(n_features());
  _features_min = ArrayDouble(n_features());
}

Node::Node(const Node &node)
    : in_tree(node.in_tree), _index(node._index), _left(node._left), _right(node._right),
      _parent(node._parent), creation_time(node.creation_time),
      feature(node.feature), threshold(node.threshold),
      impurity(node.impurity), _n_samples(node._n_samples), labels_average(node.labels_average),
      aggregation_weight(node.aggregation_weight), aggregation_weight_ctw(node.aggregation_weight_ctw),
      _features_min(node._features_min), _features_max(node._features_max),
      _is_leaf(node._is_leaf) {
  // std::cout << "Copy const of Node(index: " << _index << ")" << std::endl;
}

Node::Node(const Node &&node) : in_tree(in_tree) {
  // std::cout << "Node::Node(const Node && node)" << std::endl;
  _index = node._index;
  _left = node._left;
  _right = node._right;
  _parent = node._parent;
  creation_time = node.creation_time;
  feature = node.feature;
  threshold = node.threshold;
  impurity = node.impurity;
  _n_samples = node._n_samples;
  labels_average = node.labels_average;
  aggregation_weight = node.aggregation_weight;
  aggregation_weight_ctw = node.aggregation_weight_ctw;
  _is_leaf = node._is_leaf;
  _features_min = node._features_min;
  _features_max = node._features_max;
}

void Node::update(ulong sample_index) {
  _n_samples++;
  ArrayDouble x_t = in_tree.forest.features(sample_index);
  double y_t = in_tree.forest.label(sample_index);
  std::cout << "x_t=" << std::endl;
  x_t.print();
  std::cout << "y_t= " << y_t << std::endl;

  // TODO: j'en suis ici... mettre à jour les features, et autres calculs stat
  if (_n_samples == 1) {}
  for(ulong j=0; j < n_features(); ++j) {

  }
}

void Tree::split_node(ulong index) {
  // Choose at random the feature used to cut
  ulong left = add_node(index, iteration);
  ulong right = add_node(index, iteration);
  nodes[index].set_left(left).set_right(right).set_is_leaf(false);
}

Node &Tree::get_node(ulong node_index) {
  return nodes[node_index];
}

Tree::Tree(const Tree &tree)
    : nodes(tree.nodes), forest(tree.forest), already_fitted(tree.already_fitted) {
  std::cout << "Tree::Tree(const &Tree tree)" << std::endl;
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

void Tree::fit(ulong sample_index) {
  // TODO: Test that the size does not change within successive calls to fit
  // std::cout << "Fitting a tree" << std::endl;

  // Start at the root. Index of the root is always 0
  ulong index_current_node = 0;

  iteration++;

  bool is_leaf = false;

  // Let's go find the leaf that contains the sample
  while (!is_leaf) {
    // Get the current node
    Node &current_node = get_node(index_current_node);
    current_node.update(sample_index);
    // If the node a leaf ?
    is_leaf = current_node.is_leaf();
    if (!is_leaf) {
      if (iteration % 2 == 0) {
        index_current_node = current_node.left();
      } else {
        index_current_node = current_node.right();
      }
    }
  }
  if (nodes[index_current_node].n_samples() >= 10) {
    split_node(index_current_node);
  }
  print();
}

ulong Tree::add_node(ulong parent, ulong creation_time) {
  nodes.emplace_back(*this, n_nodes, parent, creation_time);
  return n_nodes++;
}

OnlineForest::OnlineForest(uint32_t n_trees, uint32_t n_min_samples,
                           uint32_t n_splits)
    : n_trees(n_trees), _n_min_samples(n_min_samples),
      _n_splits(n_splits), trees() {
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
      std::cout << "pass number: " << it / n_samples() << std::endl;
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
  this->_features = features;
  this->_labels = labels;
  has_data = true;
  trees.clear();
  // TODO: when we set_data, we need to recreate the trees
  for (uint32_t i = 0; i < n_trees; ++i) {
    trees.emplace_back(*this);
  }
}

inline ArrayDouble Tree::get_features(ulong sample_index) const {
  return forest.features(sample_index);
}

inline ulong Tree::n_features() const {
  return forest.get_n_features();
}

inline double Tree::get_label(ulong sample_index) const {
  return forest.label(sample_index);
}
