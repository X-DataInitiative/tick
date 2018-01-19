#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

#include <cereal/types/base_class.hpp>

class DLL_PUBLIC ModelSmoothedHinge : public virtual ModelGeneralizedLinear, public ModelLipschitz {
 private:
  double smoothness;

 public:
  ModelSmoothedHinge(const SBaseArrayDouble2dPtr features,
                     const SArrayDoublePtr labels,
                     const bool fit_intercept,
                     const double smoothness = 1.,
                     const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  void compute_lip_consts() override;

  double get_smoothness() const {
    return smoothness;
  }

  void set_smoothness(double smoothness) {
    if (smoothness <= 1e-2 || smoothness > 1) {
      TICK_ERROR("smoothness should be between 0.01 and 1");
    } else {
      this->smoothness = smoothness;
    }
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
    ar(cereal::make_nvp("ModelLipschitz", cereal::base_class<ModelLipschitz>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelSmoothedHinge, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_SMOOTHED_HINGE_H_
