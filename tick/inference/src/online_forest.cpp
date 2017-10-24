// License: BSD 3 clause

#include "online_forest.h"


Node::Node(Tree &tree) : samples(), tree(tree) {
  std::cout << "Node::Node(Tree &tree)\n";
  features_min = ArrayDouble(tree.get_forest().get_n_features());
  features_max = ArrayDouble(tree.get_forest().get_n_features());
  // At its creation, a node is a leaf
  is_leaf = true;
  // At creation of the node, there is no sample in it
  n_samples = 0;
  labels_average = 0;
  impurity = 0;
  // Initialized to one since it will receive multiplicative updates
  aggregation_weight = 1;
  aggregation_weight_ctw = 1;
}

void Node::update(ulong sample_index, bool update_range) {
  // Get the features of the sample
  ArrayDouble x_t = get_features(sample_index);
  // Get the label of the sample
  double y_t = get_label(sample_index);

  // Online update the average of the labels in the node
  labels_average = (n_samples * labels_average + y_t) / (n_samples + 1);
  // Increase the number of samples in the node
  n_samples++;

  // TODO: update aggregation_weight;
  // TODO: update aggregation_weight_ctw;

  if (update_range) {
    if (n_samples==1) {
      // This is the first sample, we build copies of the sample point
      std::cout << "creation of features_min\n";
      features_min = ArrayDouble(x_t);
      std::cout << "features_min.size()=" << features_min.size() << std::endl;
      features_max = ArrayDouble(x_t);
    } else {
      for (ulong j=0; j < x_t.size(); ++j) {
        double x_tj = x_t[j];
        if (features_min[j] > x_tj) {
          features_min[j] = x_tj;
        }
        if (features_max[j] < x_tj) {
          features_max[j] = x_tj;
        }
      }
    }
  }
  std::cout << "features_min= " << std::endl;
  features_min.print();
  std::cout << "features_max= " << std::endl;
  features_max.print();

  // Save the sample in the node
  samples.push_back(sample_index);
}


void Node::split(ulong node_index, uint32_t n_splits) {
  // Choose at random the feature used to cut

  std::cout << "get_n_features()=" << get_n_features() << std::endl;
  ulong splitting_feature = get_tree().get_forest().rand_unif(get_n_features() - 1);

  std::cout << "splitting_feature= " << splitting_feature << std::endl;
  // Choose at random the threshold in the range of the feature
  ulong cut_index = get_tree().get_forest().rand_unif(get_n_features() - 1);
  // Note that cut_index is in {0, ..., n_splits-1}
  double feature_min = features_min[splitting_feature];
  double feature_max = features_max[splitting_feature];

  std::cout << "feature_min= " << feature_min << std::endl;
  std::cout << "feature_max= " << feature_max << std::endl;
  std::cout << "cut_index= " << cut_index << std::endl;

  double threshold = feature_min + (double)(cut_index + 1) / (n_splits + 1) * (feature_max - feature_min);

  // Get the current iteration counter
  ulong t = get_tree().get_forest().get_t();

  std::cout << "t=" << t << ", threshold=" << threshold << ", feature_min="
            << feature_min << ", feature_max=" << feature_max <<  ", cut_index="
            << cut_index << ", splitting_feature=" << splitting_feature
            << ", n_splits=" << n_splits << "node_index= " << node_index << std::endl;

  // Create the childs
  // Add a left-child node, whose parent is the current node, created at iteration t
  ulong left_child = get_tree().add_node(node_index, t);
  // Add a right-child node, whose parent is the current node, created at iteration t
  ulong right_child = get_tree().add_node(node_index, t);

  // Give to the node information about its childs and how its splitting is done
  this->threshold = threshold;
  this->feature = splitting_feature;
  this->left_child = left_child;
  this->right_child = right_child;
  // This node is no more a leaf;
  this->is_leaf = false;

  std::cout << "left_child=" << left_child << ", right_child=" << right_child << std::endl;


  std::cout << "blabla= "  << std::endl;

  std::cout << "get_tree().get_node(left_child).features_min.print()= "  << std::endl;

  get_tree().get_node(left_child).get_features_min().print();

  get_tree().get_node(left_child).set_features_min(features_min);
  get_tree().get_node(left_child).set_features_max(features_max);
  get_tree().get_node(right_child).set_features_min(features_min);
  get_tree().get_node(right_child).set_features_max(features_max);

  get_tree().get_node(left_child).get_features_min().print();

  std::cout << "blabla= " << cut_index << std::endl;

  // TODO: free the memory of this node's features_min and features_max
  // get_tree().get_node(right_child).features_min[splitting_feature] = threshold;
  // get_tree().get_node(left_child).features_max[splitting_feature] = threshold;

  // Put the samples from this node into the left and right childs
  for(ulong i: samples) {
    double x_ij = get_features(i)[splitting_feature];
    if (x_ij < threshold) {
      get_tree().get_node(left_child).samples.push_back(i);
      get_tree().get_node(left_child).n_samples++;
    } else {
      get_tree().get_node(right_child).samples.push_back(i);
      get_tree().get_node(right_child).n_samples++;
    }
  }
  std::cout << "blabla= " << cut_index << std::endl;

}


Node& Tree::get_node(ulong node_index) {
  return nodes[node_index];
}

OnlineForest &Tree::get_forest() const {
  return forest;
}


double Node::get_label(ulong sample_index) const {
  return tree.get_label(sample_index);
}

ulong Node::get_n_features() const {
  return tree.get_n_features();
}


ArrayDouble Node::get_features(ulong sample_index) const {
  return tree.get_features(sample_index);
}



Tree::Tree(OnlineForest &forest) : nodes(), forest(forest) {
  std::cout << "Tree::Tree(OnlineForest &forest)\n";
  // Add the root of the Tree (first node)
  add_node();
}


void Tree::fit(ulong sample_index) {
  // TODO: Test that the size does not change within successive calls to fit
  std::cout << "Fitting a tree" << std::endl;

  // Start at the root. Index of the root is always 0
  uint32_t current_node_index = 0;

  // Let's go find the leaf that contains the sample
  while (true) {
    // Get the current node
    Node& current_node = get_node(current_node_index);
    // If the node a leaf ?
    bool is_leaf = current_node.get_is_leaf();
    // Update the node. If the node is a leaf, we update the range of the data in the node
    current_node.update(sample_index, is_leaf);
    if (is_leaf) {
      // If it's a leaf and if n_samples is large enough, split the node
      std::cout << "current_node.get_n_samples() >= n_min_samples:" << current_node.get_n_samples() << ">=" << forest.get_n_min_samples() << std::endl;
      if (current_node.get_n_samples() >= forest.get_n_min_samples()) {
        std::cout << "Splitting the node\n";
        current_node.split(current_node_index, forest.get_n_splits());
      }
      // and break
      break;
    } else {
      // Otherwise, find who the child is for this sample
      // Get the feature index used for the split
      ulong feature = current_node.get_feature();
      ArrayDouble x_t = get_features(sample_index);
      double threshold = current_node.get_threshold();
      if (x_t[feature] < threshold) {
        current_node_index = current_node.get_left_child();
      } else {
        current_node_index = current_node.get_right_child();
      }
    }
  }
}


ulong Tree::add_node() {
  nodes.emplace_back(*this);
  return nodes.size() - 1;
}

ulong Tree::add_node(uint32_t parent, ulong creation_time) {
  nodes.emplace_back(*this, parent, creation_time);
  return nodes.size() - 1;
}

OnlineForest::OnlineForest(uint32_t n_trees, uint32_t n_min_samples,
                           uint32_t n_splits)
    : n_trees(n_trees), n_min_samples(n_min_samples),
      n_splits(n_splits), trees() {

  std::cout << "Constructor of OnlineForest, n_splits=" << n_splits << std::endl;
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
    n_iter = get_n_samples();
  }
  std::cout << "n_iter=" << n_iter << std::endl;
  for (ulong it = 0; it < n_iter; ++it) {
    std::cout << "------------------" << std::endl;
    std::cout << "iteration=" << it << std::endl;
    ulong sample_index = get_next_sample();
    std::cout << "sample_index=" << sample_index << std::endl;
    for (Tree& tree : trees) {
      // Fit the tree online using the new data point
      tree.fit(sample_index);
    }
    t++;
  }
}

void OnlineForest::init_permutation() {
  if ((cycle_type== CycleType::permutation) && (get_n_samples() > 0)) {
    permutation = ArrayULong(get_n_samples());
    for (ulong i = 0; i < get_n_samples(); ++i)
      permutation[i] = i;
  }
}


//// Simulation of a random permutation using Knuth's algorithm
void OnlineForest::shuffle() {
  if (cycle_type == CycleType::permutation) {
    // A secure check
    if (permutation.size() != get_n_samples()) {
      init_permutation();
    }
    // Restart the i_perm
    i_perm = 0;
    for (ulong i = 1; i < get_n_samples(); ++i) {
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
    i = rand_unif(get_n_samples() - 1);
  } else {
    if (cycle_type == CycleType::permutation) {
      if (!permutation_ready) {
        shuffle();
      }
      i = permutation[i_perm];
      i_perm++;
      if (i_perm >= get_n_samples()) {
        shuffle();
      }
    } else {
      // Otherwise it's cycling through the data
      i = i_perm;
      i_perm++;
      if (i_perm >= get_n_samples()) {
        i_perm = 0;
      }
    }
  }
  return i;
}

void OnlineForest::set_data(const SArrayDouble2dPtr features,
                            const SArrayDoublePtr labels) {
  this->features = features;
  this->labels = labels;
  has_data = true;
  trees.clear();
  // TODO: when we set_data, we need to recreate the trees
  for (uint32_t i = 0; i < n_trees; ++i) {
    trees.emplace_back(*this);
  }
}


inline ArrayDouble Tree::get_features(ulong sample_index) const {
  return forest.get_features(sample_index);
}


inline ulong Tree::get_n_features() const {
  return forest.get_n_features();
}

inline double Tree::get_label(ulong sample_index) const {
  return forest.get_label(sample_index);
}
