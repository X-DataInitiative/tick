
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"

#include <cereal/types/base_class.hpp>

class DLL_PUBLIC ModelAbsoluteRegression : public virtual ModelGeneralizedLinear {
 public:
  ModelAbsoluteRegression(const SBaseArrayDouble2dPtr features,
                          const SArrayDoublePtr labels,
                          const bool fit_intercept,
                          const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelAbsoluteRegression, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_ABSOLUTE_REGRESSION_H_
