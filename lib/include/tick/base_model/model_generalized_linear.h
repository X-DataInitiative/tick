//
// Created by Stéphane GAIFFAS on 06/12/2015.
//
#ifndef LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
#define LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_

// License: BSD 3 clause

#include "model_labels_features.h"

class DLL_PUBLIC ModelGeneralizedLinear : public ModelLabelsFeatures {
 protected:
  ArrayDouble features_norm_sq;

  unsigned int n_threads;

  bool fit_intercept;

  /**
   * Computes gradient fo ith observation
   * @param i : The selected observation
   * @param out : Preallocated vector in which information is store
   * @param coeffs : coefficient at which the gradient is computed
   * @param fill : If `true` out will be filled by the gradient value, otherwise out will be
   * inceremented by the gradient value.
   */
  virtual void compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                              ArrayDouble &out, const bool fill);

    bool ready_features_norm_sq;

    void compute_features_norm_sq();

 public:
  ModelGeneralizedLinear(const SBaseArrayDouble2dPtr features,
                         const SArrayDoublePtr labels,
                         const bool fit_intercept,
                         const int n_threads = 1);

  const char *get_class_name() const override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * To be used by grad(ArrayDouble&, ArrayDouble&) to calculate grad by incrementally
   * updating 'out'
   * out and coeffs are not in the same order as in grad_i as this is necessary for
   * parallel_map_array
   */
  virtual void inc_grad_i(const ulong i, ArrayDouble &out, const ArrayDouble &coeffs);

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  double loss(const ArrayDouble &coeffs) override;

  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector) override;

  bool use_intercept() const override {
    return fit_intercept;
  }

  bool is_sparse() const override {
    return features->is_sparse();
  }

  ulong get_n_coeffs() const override {
    return get_n_features() + static_cast<int>(fit_intercept);
  }

  virtual double get_inner_prod(const ulong i, const ArrayDouble &coeffs) const;

  virtual void set_fit_intercept(const bool fit_intercept) {
    this->fit_intercept = fit_intercept;
  }

  virtual bool get_fit_intercept() const {
    return fit_intercept;
  }

  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ModelLabelsFeatures", cereal::base_class<ModelLabelsFeatures>(this)));
    ar(CEREAL_NVP(features_norm_sq));
    ar(CEREAL_NVP(fit_intercept));
    ar(CEREAL_NVP(ready_features_norm_sq));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelGeneralizedLinear, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_BASE_MODEL_MODEL_GENERALIZED_LINEAR_H_
