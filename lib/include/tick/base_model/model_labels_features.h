
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_

// License: BSD 3 clause

#include "model.h"

#include <iostream>

class DLL_PUBLIC ModelLabelsFeatures : public virtual Model {
 protected:
  ulong n_samples, n_features;

  //! Labels vector
  SArrayDoublePtr labels;

  //! Features matrix (either sparse or not)
  SBaseArrayDouble2dPtr features;

  bool ready_columns_sparsity;
  ArrayDouble column_sparsity;

 public:
  ModelLabelsFeatures(SBaseArrayDouble2dPtr features,
                      SArrayDoublePtr labels);

  const char *get_class_name() const override {
    return "ModelLabelsFeatures";
  }

  ulong get_n_samples() const override {
    return n_samples;
  }

  ulong get_n_features() const override {
    return n_features;
  }

  // TODO: add consts
  BaseArrayDouble get_features(ulong i) const override {
    return view_row(*features, i);
  }

  virtual double get_label(ulong i) const {
    return (*labels)[i];
  }

  virtual ulong get_rand_max() const {
    return n_samples;
  }

  ulong get_epoch_size() const override {
    return n_samples;
  }

  bool is_ready_columns_sparsity() const {
    return ready_columns_sparsity;
  }

  ArrayDouble get_column_sparsity_view() {
    if (!is_ready_columns_sparsity()) compute_columns_sparsity();
    return view(column_sparsity);
  }

  void compute_columns_sparsity();

  template<class Archive>
  void load(Archive &ar) {
    ar(CEREAL_NVP(n_samples));

    ArrayDouble temp_labels;
    ArrayDouble2d temp_features;
    ar(cereal::make_nvp("labels", temp_labels));
    ar(cereal::make_nvp("features", temp_features));

    labels = temp_labels.as_sarray_ptr();
    features = temp_features.as_sarray2d_ptr();
  }

  template<class Archive>
  void save(Archive &ar) const {
    ar(CEREAL_NVP(n_samples));
    ar(cereal::make_nvp("labels", *labels));
    ar(cereal::make_nvp("features", *features));
  }
};

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_LABELS_FEATURES_H_
