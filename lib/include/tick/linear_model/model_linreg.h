//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

#include <cereal/types/base_class.hpp>

template <class T, class K = T>
class DLL_PUBLIC TModelLinReg : public virtual TModelGeneralizedLinear<T, K>, public TModelLipschitz<T, K> {
 private:
  std::string clazz = "TModelGeneralizedLinearWithIntercepts<"
    + std::string(typeid(T).name())
    + ", " + std::string(typeid(K).name()) + ">";

 protected:
  using TModelLipschitz<T, K>::ready_lip_consts;
  using TModelLipschitz<T, K>::lip_consts;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::fit_intercept;

 public:
  using TModelGeneralizedLinear<T, K>::get_label;
  using TModelGeneralizedLinear<T, K>::use_intercept;
  using TModelGeneralizedLinear<T, K>::get_inner_prod;

 public:
  TModelLinReg(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1);

  virtual ~TModelLinReg() {}

  virtual const char *get_class_name() const {
    return clazz.c_str();
  }

  K sdca_dual_min_i(const ulong i,
                         const K dual_i,
                         const Array<K> &primal_vector,
                         const K previous_delta_dual_i,
                         K l_l2sq) override;

  K loss_i(const ulong i, const Array<T> &coeffs) override;

  K grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  void compute_lip_consts() override;

  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", typename cereal::virtual_base_class<TModelGeneralizedLinear<T, K> >(this)));
    ar(cereal::make_nvp("ModelLipschitz", typename cereal::base_class<TModelLipschitz<T, K> >(this)));
  }
};

class DLL_PUBLIC ModelLinReg : public TModelLinReg<double, double> {
 public:
  ModelLinReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads = 1);

  const char *get_class_name() const override;
};

using ModelLinRegDouble = TModelLinReg<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegDouble, cereal::specialization::member_serialize)

using ModelLinRegFloat = TModelLinReg<float , float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinRegFloat, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LINREG_H_
