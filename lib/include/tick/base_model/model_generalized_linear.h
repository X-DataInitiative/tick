//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_

// License: BSD 3 clause

#include "model_labels_features.h"

template <class T, class K = T>
class DLL_PUBLIC TModelGeneralizedLinear
    : virtual public TModelLabelsFeatures<T, K> {

 public:
  using TModelLabelsFeatures<T, K>::get_label;
  using TModelLabelsFeatures<T, K>::get_n_features;
  using TModelLabelsFeatures<T, K>::get_n_samples;
  using TModelLabelsFeatures<T, K>::get_class_name;

 protected:
  using TModelLabelsFeatures<T, K>::features;
  using TModelLabelsFeatures<T, K>::labels;
  using TModelLabelsFeatures<T, K>::n_samples;
  using TModelLabelsFeatures<T, K>::n_features;
  using TModelLabelsFeatures<T, K>::get_features;
  using TModelLabelsFeatures<T, K>::is_ready_columns_sparsity;

  bool fit_intercept = false;
  bool ready_features_norm_sq = false;

  unsigned int n_threads = 0;

  Array<T> features_norm_sq;

  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise
   * out will be inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const Array<K> &coeffs,
                              Array<T> &out, const bool fill);

  Array<T> &get_features_norm_sq() { return features_norm_sq; }

 public:
  void compute_features_norm_sq();

  T get_features_norm_sq_i(ulong i) { return features_norm_sq[i]; }

 public:
  TModelGeneralizedLinear(const std::shared_ptr<BaseArray2d<T> > features,
                          const std::shared_ptr<SArray<T> > labels,
                          const bool fit_intercept, const int n_threads = 1);

  virtual ~TModelGeneralizedLinear() {}

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  void grad_i(const ulong i, const Array<K> &coeffs, Array<T> &out) override;

  /**
   * To be used by grad(ArrayDouble&, ArrayDouble&) to calculate grad by
   * incrementally updating 'out' out and coeffs are not in the same order as in
   * grad_i as this is necessary for parallel_map_array
   */
  virtual void inc_grad_i(const ulong i, Array<T> &out, const Array<K> &coeffs);

  void grad(const Array<K> &coeffs, Array<T> &out) override;

  T loss(const Array<K> &coeffs) override;

  void sdca_primal_dual_relation(const T l_l2sq, const Array<K> &dual_vector,
                                 Array<K> &out_primal_vector) override;

  bool use_intercept() const override { return fit_intercept; }

  bool is_sparse() const override { return features->is_sparse(); }

  ulong get_n_coeffs() const override {
    return get_n_features() + static_cast<int>(fit_intercept);
  }

  virtual T get_inner_prod(const ulong i, const Array<K> &coeffs) const;

  virtual void set_fit_intercept(const bool fit_intercept) {
    this->fit_intercept = fit_intercept;
  }

  virtual bool get_fit_intercept() const { return fit_intercept; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelLabelsFeatures",
        typename cereal::virtual_base_class<TModelLabelsFeatures<T, K> >(
            this)));

    ar(CEREAL_NVP(features_norm_sq));
    ar(CEREAL_NVP(fit_intercept));
    ar(CEREAL_NVP(ready_features_norm_sq));
    ar(CEREAL_NVP(n_threads));
  }

 protected:
  BoolStrReport compare(const TModelGeneralizedLinear<T, K> &that,
                        std::stringstream &ss) {
    auto are_equal = TModelLabelsFeatures<T, K>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, features_norm_sq) &&
                     TICK_CMP_REPORT(ss, fit_intercept) &&
                     TICK_CMP_REPORT(ss, n_features) &&
                     TICK_CMP_REPORT(ss, ready_features_norm_sq) &&
                     TICK_CMP_REPORT(ss, n_threads);
    return BoolStrReport(are_equal, ss.str());
  }
};

using ModelGeneralizedLinear = TModelGeneralizedLinear<double, double>;

using ModelGeneralizedLinearDouble = TModelGeneralizedLinear<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearDouble,
                                   cereal::specialization::member_serialize)

using ModelGeneralizedLinearFloat = TModelGeneralizedLinear<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearFloat,
                                   cereal::specialization::member_serialize)

using ModelGeneralizedLinearAtomicDouble =
    TModelGeneralizedLinear<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearAtomicDouble,
                                   cereal::specialization::member_serialize)

using ModelGeneralizedLinearAtomicFloat =
    TModelGeneralizedLinear<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearAtomicFloat,
                                   cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
