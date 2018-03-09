//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_

// License: BSD 3 clause

#include "model_labels_features.h"

template <class T>
class DLL_PUBLIC TModelGeneralizedLinear
    : virtual public TModelLabelsFeatures<T> {
 protected:
  using TModelLabelsFeatures<T>::features;
  using TModelLabelsFeatures<T>::labels;
  using TModelLabelsFeatures<T>::n_samples;
  using TModelLabelsFeatures<T>::get_n_samples;
  using TModelLabelsFeatures<T>::n_features;
  using TModelLabelsFeatures<T>::get_n_features;
  using TModelLabelsFeatures<T>::get_label;
  using TModelLabelsFeatures<T>::get_features;
  using TModelLabelsFeatures<T>::is_ready_columns_sparsity;
  using TModelLabelsFeatures<T>::get_class_name;

  bool fit_intercept = false;
  bool ready_features_norm_sq = false;

  unsigned int n_threads;

  Array<T> features_norm_sq;

  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise
   * out will be inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const Array<T> &coeffs,
                              Array<T> &out, const bool fill);

  void compute_features_norm_sq();

  Array<T> &get_features_norm_sq() { return features_norm_sq; }

 public:
  TModelGeneralizedLinear(const std::shared_ptr<BaseArray2d<T> > features,
                          const std::shared_ptr<SArray<T> > labels,
                          const bool fit_intercept, const int n_threads = 1);

  virtual ~TModelGeneralizedLinear() {}

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  void grad_i(const ulong i, const Array<T> &coeffs, Array<T> &out) override;

  /**
   * To be used by grad(ArrayDouble&, ArrayDouble&) to calculate grad by
   * incrementally updating 'out' out and coeffs are not in the same order as in
   * grad_i as this is necessary for parallel_map_array
   */
  virtual void inc_grad_i(const ulong i, Array<T> &out, const Array<T> &coeffs);

  void grad(const Array<T> &coeffs, Array<T> &out) override;

  T loss(const Array<T> &coeffs) override;

  void sdca_primal_dual_relation(const T l_l2sq, const Array<T> &dual_vector,
                                 Array<T> &out_primal_vector) override;

  bool use_intercept() const override { return fit_intercept; }

  bool is_sparse() const override { return features->is_sparse(); }

  ulong get_n_coeffs() const override {
    return get_n_features() + static_cast<int>(fit_intercept);
  }

  virtual T get_inner_prod(const ulong i, const Array<T> &coeffs) const;

  virtual void set_fit_intercept(const bool fit_intercept) {
    this->fit_intercept = fit_intercept;
  }

  virtual bool get_fit_intercept() const { return fit_intercept; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelLabelsFeatures",
                        cereal::base_class<TModelLabelsFeatures<T> >(this)));
    ar(CEREAL_NVP(features_norm_sq));
    ar(CEREAL_NVP(fit_intercept));
    ar(CEREAL_NVP(ready_features_norm_sq));
    ar(CEREAL_NVP(n_threads));
  }

 protected:
  BoolStrReport compare(const TModelGeneralizedLinear<T> &that,
                        std::stringstream &ss) {
    auto are_equal = TModelLabelsFeatures<T>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, features_norm_sq) &&
                     TICK_CMP_REPORT(ss, fit_intercept) &&
                     TICK_CMP_REPORT(ss, n_features) &&
                     TICK_CMP_REPORT(ss, ready_features_norm_sq) &&
                     TICK_CMP_REPORT(ss, n_threads);
    return BoolStrReport(are_equal, ss.str());
  }
};

using ModelGeneralizedLinear = TModelGeneralizedLinear<double>;

using ModelGeneralizedLinearDouble = TModelGeneralizedLinear<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearDouble,
                                   cereal::specialization::member_serialize)

using ModelGeneralizedLinearFloat = TModelGeneralizedLinear<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearFloat,
                                   cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
