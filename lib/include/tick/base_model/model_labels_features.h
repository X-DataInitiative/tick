
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_

// License: BSD 3 clause

#include "model.h"

#include <string>

template <class T>
class DLL_PUBLIC TModelLabelsFeatures : public virtual TModel<T> {
 protected:
  bool ready_columns_sparsity;
  ulong n_samples, n_features;

  //! Labels vector
  std::shared_ptr<SArray<T> > labels;

  //! Features matrix (either sparse or not)
  std::shared_ptr<BaseArray2d<T> > features;

  Array<T> column_sparsity;

 public:
  TModelLabelsFeatures(const std::shared_ptr<BaseArray2d<T> > features,
                       const std::shared_ptr<SArray<T> > labels);
  TModelLabelsFeatures(const TModelLabelsFeatures &) = delete;
  TModelLabelsFeatures(const TModelLabelsFeatures &&) = delete;

  virtual ~TModelLabelsFeatures() {}

  ulong get_n_samples() const override { return n_samples; }

  ulong get_n_features() const override { return n_features; }

  // TODO: add consts
  BaseArray<T> get_features(ulong i) const override {
    return view_row(*features, i);
  }

  virtual T get_label(ulong i) const { return (*labels)[i]; }

  virtual ulong get_rand_max() const { return n_samples; }

  ulong get_epoch_size() const override { return n_samples; }

  bool is_ready_columns_sparsity() const { return ready_columns_sparsity; }

  Array<T> get_column_sparsity_view() {
    if (!is_ready_columns_sparsity()) compute_columns_sparsity();
    return view(column_sparsity);
  }

  void compute_columns_sparsity();

  template <class Archive>
  void load(Archive &ar) {
    ar(cereal::make_nvp("Model", cereal::base_class<TModel<T> >(this)));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(ready_columns_sparsity));
    ar(CEREAL_NVP(column_sparsity));

    Array<T> temp_labels;
    Array2d<T> temp_features;
    ar(cereal::make_nvp("labels", temp_labels));
    ar(cereal::make_nvp("features", temp_features));

    labels = temp_labels.as_sarray_ptr();
    features = temp_features.as_sarray2d_ptr();
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("Model", cereal::base_class<TModel<T> >(this)));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(ready_columns_sparsity));
    ar(CEREAL_NVP(column_sparsity));

    ar(cereal::make_nvp("labels", *labels));
    ar(cereal::make_nvp("features", *features));
  }

 protected:
  BoolStrReport compare(const TModelLabelsFeatures<T> &that,
                        std::stringstream &ss) {
    bool are_equal =
        TICK_CMP_REPORT(ss, ready_columns_sparsity) &&
        TICK_CMP_REPORT(ss, n_samples) && TICK_CMP_REPORT(ss, n_features) &&
        TICK_CMP_REPORT(ss, column_sparsity) &&
        TICK_CMP_REPORT_PTR(ss, features) && TICK_CMP_REPORT_PTR(ss, labels);
    return BoolStrReport(are_equal, ss.str());
  }
};

using ModelLabelsFeatures = TModelLabelsFeatures<double>;

using ModelLabelsFeaturesDouble = TModelLabelsFeatures<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesDouble,
                                   cereal::specialization::member_load_save);

using ModelLabelsFeaturesFloat = TModelLabelsFeatures<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesFloat,
                                   cereal::specialization::member_load_save);

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
