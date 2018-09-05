// License: BSD 3 clause

%include std_shared_ptr.i
%shared_ptr(OnlineForestRegressor);

%{
#include "tick/online/online_forest_classifier.h"
#include "tick/online/online_forest_classifier.h"
%}


enum class CriterionClassifier {
  log = 0
};


enum class FeatureImportanceType {
  no = 0,
  estimated = 1,
  given = 2
};


class OnlineForestClassifier {
 public:

  OnlineForestClassifier(uint32_t n_features,
                         uint8_t n_classes,
                         uint8_t n_trees,
                         float step,
                         CriterionClassifier criterion,
                         FeatureImportanceType feature_importance_type,
                         bool use_aggregation,
                         float dirichlet,
                         bool split_pure,
                         int32_t max_nodes,
                         float min_extension_size,
                         int32_t min_samples_split,
                         int32_t max_features,
                         int32_t n_threads,
                         int seed,
                         bool verbose,
                         uint32_t print_every,
                         uint32_t max_nodes_with_memory);

  void fit(const SArrayFloat2dPtr features, const SArrayFloatPtr labels);
  void predict(const SArrayFloat2dPtr features, SArrayFloat2dPtr predictions);

  void clear();

  void print();

  uint32_t n_samples() const;
  uint32_t n_features() const;
  uint8_t n_classes() const;

  inline float step() const;
  inline OnlineForestClassifier& step(const float step);

  uint32_t n_trees() const;

  int32_t n_threads() const;
  OnlineForestClassifier &n_threads(int32_t n_threads);

  CriterionClassifier criterion() const;
  OnlineForestClassifier &criterion(CriterionClassifier criterion);

  int seed() const;
  OnlineForestClassifier &seed(int seed);

  void n_nodes(SArrayUIntPtr n_nodes_per_tree);
  void n_nodes_reserved(SArrayUIntPtr n_reserved_nodes_per_tree);
  void n_leaves(SArrayUIntPtr n_leaves_per_tree);

  bool verbose() const;
  OnlineForestClassifier &verbose(bool verbose);

  void get_feature_importances(SArrayFloatPtr feature_importances);

  OnlineForestClassifier &given_feature_importances(const ArrayFloat &feature_importances);

  uint32_t get_path_depth(const uint8_t tree, const SArrayFloatPtr x_t);

  void get_path(const uint8_t tree, const SArrayFloatPtr x_t, SArrayUIntPtr path);

  void get_flat_nodes(
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
      SArrayUInt2dPtr nodes_counts
  );

};
