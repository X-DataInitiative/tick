
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

#include <cereal/types/base_class.hpp>

template <class T>
class DLL_PUBLIC TModelAbsoluteRegression
    : public virtual TModelGeneralizedLinear<T> {
 protected:
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::n_features;
  using TModelGeneralizedLinear<T>::fit_intercept;
  using TModelGeneralizedLinear<T>::compute_grad_i;
  using TModelGeneralizedLinear<T>::n_threads;
  using TModelGeneralizedLinear<T>::get_inner_prod;

 public:
  using TModelGeneralizedLinear<T>::get_label;
  using TModelGeneralizedLinear<T>::grad_i;
  using TModelGeneralizedLinear<T>::get_features;
  using TModelGeneralizedLinear<T>::grad_i_factor;

 public:
  TModelAbsoluteRegression(const std::shared_ptr<BaseArray2d<T> > features,
                           const std::shared_ptr<SArray<T> > labels,
                           const bool fit_intercept, const int n_threads = 1);

  const char *get_class_name() const override;

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear",
                        cereal::base_class<ModelGeneralizedLinear>(this)));
  }
};

using ModelAbsoluteRegression = TModelAbsoluteRegression<double>;

using ModelAbsoluteRegressionDouble = TModelAbsoluteRegression<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelAbsoluteRegressionDouble,
                                   cereal::specialization::member_serialize)

using ModelAbsoluteRegressionFloat = TModelAbsoluteRegression<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelAbsoluteRegressionFloat,
                                   cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_
