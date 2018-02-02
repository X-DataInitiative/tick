// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForestRegressor);

%{
#include "tick/online/online_forest_classifier.h"
%}


enum class CriterionClassifier {
  log = 0
};


enum class FeatureImportanceType {
  no = 0,
  estimated,
  given
};


class OnlineForestClassifier {
 public:

  OnlineForestClassifier(uint32_t n_features, uint8_t n_classes, uint8_t n_trees,
                         uint8_t n_passes = 1, double step = 1.0,
                         CriterionClassifier criterion = CriterionClassifier::log,
                         FeatureImportanceType feature_importance_type = FeatureImportanceType::estimated,
                         bool use_aggregation = true, double subsampling=1, double dirichlet=0.5,
                         int32_t n_threads = 1, int seed = 0, bool verbose = false);

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDouble2dPtr predictions);

  void clear();

  void print();

  uint32_t n_samples() const;
  uint32_t n_features() const;
  uint8_t n_classes() const;

  inline double step() const;
  inline OnlineForestClassifier& set_step(const double step);

  uint32_t n_trees() const;

  int32_t n_threads() const;
  OnlineForestClassifier &set_n_threads(int32_t n_threads);

  CriterionClassifier criterion() const;
  OnlineForestClassifier &set_criterion(CriterionClassifier criterion);

  int seed() const;
  OnlineForestClassifier &set_seed(int seed);

  void n_nodes(SArrayUIntPtr n_nodes_per_tree);
  void n_leaves(SArrayUIntPtr n_leaves_per_tree);

  bool verbose() const;
  OnlineForestClassifier &set_verbose(bool verbose);

  void get_feature_importances(SArrayDoublePtr feature_importances);

  OnlineForestClassifier &set_given_feature_importances(const ArrayDouble &feature_importances);

};
