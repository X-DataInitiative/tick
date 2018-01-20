
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

#include <cereal/types/base_class.hpp>

class DLL_PUBLIC ModelHuber : public virtual ModelGeneralizedLinear, public ModelLipschitz {
 private:
  double threshold, threshold_squared_over_two;

 public:
  ModelHuber(const SBaseArrayDouble2dPtr features,
             const SArrayDoublePtr labels,
             const bool fit_intercept,
             const double threshold,
             const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  void compute_lip_consts() override;

  virtual double get_threshold(void) const {
    return threshold;
  }

  virtual void set_threshold(const double threshold) {
    if (threshold <= 0.) {
      TICK_ERROR("threshold must be > 0");
    } else {
      this->threshold = threshold;
      threshold_squared_over_two = threshold * threshold / 2;
    }
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
    ar(cereal::make_nvp("ModelLipschitz", cereal::base_class<ModelLipschitz>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHuber, cereal::specialization::member_serialize)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_
