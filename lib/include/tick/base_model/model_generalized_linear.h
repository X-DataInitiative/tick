//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_

// License: BSD 3 clause

#include "model_labels_features.h"

template <class T, class K = T>
class DLL_PUBLIC TModelGeneralizedLinear : virtual public TModelLabelsFeatures<T, K> {
 private:
  std::string clazz = "TModelLabelsFeatures<"
    + std::string(typeid(T).name())
    + ", " + std::string(typeid(K).name()) + ">";

 protected:
  using TModelLabelsFeatures<T, K>::features;
  using TModelLabelsFeatures<T, K>::labels;
  using TModelLabelsFeatures<T, K>::n_samples;
  using TModelLabelsFeatures<T, K>::get_n_samples;
  using TModelLabelsFeatures<T, K>::n_features;
  using TModelLabelsFeatures<T, K>::get_n_features;
  using TModelLabelsFeatures<T, K>::get_label;
  using TModelLabelsFeatures<T, K>::get_features;
  using TModelLabelsFeatures<T, K>::is_ready_columns_sparsity;

  bool fit_intercept = 0;
  bool ready_features_norm_sq = 0;

  unsigned int n_threads;

  Array<K> features_norm_sq;

  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise out will be
   * inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const Array<T> &coeffs,
                              Array<K> &out, const bool fill);

  void compute_features_norm_sq();

  Array<K>& get_features_norm_sq() {
    return features_norm_sq;
  }

 public:
  TModelGeneralizedLinear(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1);

  virtual ~TModelGeneralizedLinear() {}

  virtual const char *get_class_name() const {
    return clazz.c_str();
  }

  virtual K grad_i_factor(const ulong i, const Array<T> &coeffs);

  virtual void grad_i(const ulong i, const Array<T> &coeffs, Array<K> &out);

  /**
   * To be used by grad(ArrayDouble&, ArrayDouble&) to calculate grad by incrementally
   * updating 'out'
   * out and coeffs are not in the same order as in grad_i as this is necessary for
   * parallel_map_array
   */
  virtual void inc_grad_i(const ulong i, Array<K> &out, const Array<T> &coeffs);

  virtual void grad(const Array<T> &coeffs, Array<K> &out);

  virtual K loss(const Array<T> &coeffs);

  void sdca_primal_dual_relation(const K l_l2sq,
                                 const Array<K> &dual_vector,
                                 Array<K> &out_primal_vector) override;

  bool use_intercept() const override {
    return fit_intercept;
  }

  bool is_sparse() const override {
    return features->is_sparse();
  }

  ulong get_n_coeffs() const override {
    return get_n_features() + static_cast<int>(fit_intercept);
  }

  virtual K get_inner_prod(const ulong i, const Array<T> &coeffs) const;

  virtual void set_fit_intercept(const bool fit_intercept) {
    this->fit_intercept = fit_intercept;
  }

  virtual bool get_fit_intercept() const {
    return fit_intercept;
  }

  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ModelLabelsFeatures", typename cereal::virtual_base_class<TModelLabelsFeatures<T, K> >(this)));
    ar(CEREAL_NVP(features_norm_sq));
    ar(CEREAL_NVP(fit_intercept));
    ar(CEREAL_NVP(ready_features_norm_sq));
  }
};

class DLL_PUBLIC ModelGeneralizedLinear : public TModelGeneralizedLinear<double, double> {
 protected:
  using TModelGeneralizedLinear<double, double>::features_norm_sq;
  using TModelGeneralizedLinear<double, double>::compute_features_norm_sq;
  using TModelGeneralizedLinear<double, double>::n_samples;
  using TModelGeneralizedLinear<double, double>::grad;

 public:
  ModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);

  const char *get_class_name() const override;
};

using ModelGeneralizedLinearDouble = TModelGeneralizedLinear<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearDouble, cereal::specialization::member_serialize)

using ModelGeneralizedLinearFloat  = TModelGeneralizedLinear<float , float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinearFloat, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
