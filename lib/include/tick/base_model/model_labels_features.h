
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_

// License: BSD 3 clause

#include "model.h"
#include "tick/array/sarray2d.h"
#include "tick/array/ssparsearray2d.h"

template <class T, class K = T>
class DLL_PUBLIC TModelLabelsFeatures : public virtual TModel<T, K> {
 protected:
  bool ready_columns_sparsity = false;
  ulong n_samples = 0, n_features = 0;

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
    ar(cereal::make_nvp("Model", cereal::base_class<TModel<T, K> >(this)));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(ready_columns_sparsity));
    ar(CEREAL_NVP(column_sparsity));

    bool has_labels = false;
    ar(CEREAL_NVP(has_labels));
    if (has_labels) {
      Array<T> serialized_labels;
      ar(cereal::make_nvp("labels", serialized_labels));
      labels = SArray<T>::new_ptr(serialized_labels);
    } else {
      labels = nullptr;
    }

    bool has_features = false;
    ar(CEREAL_NVP(has_features));
    if (has_features) {
      bool features_are_sparse = false;
      ar(CEREAL_NVP(features_are_sparse));
      if (features_are_sparse) {
        SparseArray2d<T> serialized_features;
        ar(cereal::make_nvp("features", serialized_features));
        features = SSparseArray2d<T>::new_ptr(serialized_features);
      } else {
        Array2d<T> serialized_features;
        ar(cereal::make_nvp("features", serialized_features));
        features = SArray2d<T>::new_ptr(serialized_features);
      }
    } else {
      features = nullptr;
    }
  }

  template <class Archive>
  void save(Archive &ar) const {
    ar(cereal::make_nvp("Model", cereal::base_class<TModel<T, K> >(this)));
    ar(CEREAL_NVP(n_samples));
    ar(CEREAL_NVP(n_features));
    ar(CEREAL_NVP(ready_columns_sparsity));
    ar(CEREAL_NVP(column_sparsity));

    const bool has_labels = labels != nullptr;
    ar(CEREAL_NVP(has_labels));
    if (has_labels) {
      const Array<T> serialized_labels(*labels);
      ar(cereal::make_nvp("labels", serialized_labels));
    }

    const bool has_features = features != nullptr;
    ar(CEREAL_NVP(has_features));
    if (has_features) {
      const bool features_are_sparse = features->is_sparse();
      ar(CEREAL_NVP(features_are_sparse));
      if (features_are_sparse) {
        const auto &serialized_features =
            static_cast<const SparseArray2d<T> &>(*features);
        ar(cereal::make_nvp("features", serialized_features));
      } else {
        const auto &serialized_features =
            static_cast<const Array2d<T> &>(*features);
        ar(cereal::make_nvp("features", serialized_features));
      }
    }
  }

 protected:
  BoolStrReport compare(const TModelLabelsFeatures<T, K> &that,
                        std::stringstream &ss) {
    bool are_equal =
        TICK_CMP_REPORT(ss, ready_columns_sparsity) &&
        TICK_CMP_REPORT(ss, n_samples) && TICK_CMP_REPORT(ss, n_features) &&
        TICK_CMP_REPORT(ss, column_sparsity) &&
        TICK_CMP_REPORT_PTR(ss, features) && TICK_CMP_REPORT_PTR(ss, labels);
    return BoolStrReport(are_equal, ss.str());
  }
};

using ModelLabelsFeatures = TModelLabelsFeatures<double, double>;

using ModelLabelsFeaturesDouble = TModelLabelsFeatures<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesDouble,
                                   cereal::specialization::member_load_save);

using ModelLabelsFeaturesFloat = TModelLabelsFeatures<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesFloat,
                                   cereal::specialization::member_load_save);

using ModelLabelsFeaturesAtomicDouble =
    TModelLabelsFeatures<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesAtomicDouble,
                                   cereal::specialization::member_load_save);

using ModelLabelsFeaturesAtomicFloat =
    TModelLabelsFeatures<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLabelsFeaturesAtomicFloat,
                                   cereal::specialization::member_load_save);

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
