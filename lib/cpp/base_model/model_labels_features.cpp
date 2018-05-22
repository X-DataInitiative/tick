// License: BSD 3 clause

#include "tick/base_model/model_labels_features.h"

template <class T, class K>
TModelLabelsFeatures<T, K>::TModelLabelsFeatures(
    const std::shared_ptr<BaseArray2d<T>> features,
    const std::shared_ptr<SArray<T>> labels)
    : ready_columns_sparsity(false),
      n_samples(labels.get() ? labels->size() : 0),
      n_features(features.get() ? features->n_cols() : 0),
      labels(labels),
      features(features) {
  if (labels.get() && labels->size() != features->n_rows()) {
    std::stringstream ss;
    ss << "In ModelLabelsFeatures, number of labels is " << labels->size();
    ss << " while the features matrix has " << features->n_rows() << " rows.";
    throw std::invalid_argument(ss.str());
  }
}

template <class T, class K>
void TModelLabelsFeatures<T, K>::compute_columns_sparsity() {
  if (features->is_sparse()) {
    column_sparsity = Array<T>(n_features);
    column_sparsity.fill(0.);
    for (ulong i = 0; i < n_samples; ++i) {
      BaseArray<T> features_i = get_features(i);
      for (ulong j = 0; j < features_i.size_sparse(); ++j) {
        // Even if the entry is zero (nothing forbids to store zeros...)
        // increment the number of non-zeros of the columns. This is necessary
        // when computed step-size corrections used in probabilistic updates
        // (see SVRG::solve_sparse_proba_updates code for instance)
        column_sparsity[features_i.indices()[j]] += 1;
      }
    }
    column_sparsity.multiply(1. / n_samples);
    ready_columns_sparsity = true;
  } else {
    TICK_ERROR("The features matrix is not sparse.")
  }
}

template class TModelLabelsFeatures<double, double>;
template class TModelLabelsFeatures<float, float>;

template class TModelLabelsFeatures<double, std::atomic<double>>;
template class TModelLabelsFeatures<float, std::atomic<float>>;
