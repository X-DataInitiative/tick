// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForestRegressor);

%{
#include "online_forest_classifier.h"
%}


enum class CriterionClassifier {
  log = 0
};

class OnlineForestClassifier {
 public:

  OnlineForestClassifier(uint8_t n_classes, uint32_t n_trees, uint8_t n_passes = 1, double step = 1.0,
                         CriterionClassifier criterion = CriterionClassifier::log,
                         bool use_aggregation = true, double subsampling=1, double dirichlet=0.5,
                         int32_t n_threads = 1, int seed = 0, bool verbose = false);

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDouble2dPtr predictions);

  void clear();

  inline double step() const;
  void print();

  ulong n_samples() const;
  ulong n_features() const;
  uint8_t n_classes() const;
  OnlineForestClassifier & set_n_classes(uint8_t n_classes);

  ulong n_classes() const;
  // OnlineForestClassifier &set_n_features(ulong n_features);

  uint32_t n_trees() const;
  OnlineForestClassifier &set_n_trees(uint32_t n_trees);

  int32_t n_threads() const;
  OnlineForestClassifier &set_n_threads(int32_t n_threads);
  CriterionClassifier criterion() const;
  OnlineForestClassifier &set_criterion(CriterionClassifier criterion);
  int seed() const;
  OnlineForestClassifier &set_seed(int seed);

  void n_nodes(SArrayUIntPtr n_nodes_per_tree);
  void n_leaves(SArrayUIntPtr n_leaves_per_tree);

  // bool verbose() const;
  // OnlineForestRegressor &set_verbose(bool verbose);

  void set_feature_importances(const ArrayDouble &feature_importances);
};
