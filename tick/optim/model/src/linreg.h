//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#ifndef TICK_OPTIM_MODEL_SRC_LINREG_H_
#define TICK_OPTIM_MODEL_SRC_LINREG_H_

#include "model_generalized_linear.h"
#include "model_lipschitz.h"

#include <cereal/types/base_class.hpp>

class ModelLinReg : public ModelGeneralizedLinear, public ModelLipschitz {
 public:
  ModelLinReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads = 1);

  const char *get_class_name() const override;

  double sdca_dual_min_i(const ulong i,
                         const ArrayDouble &dual_vector,
                         const ArrayDouble &primal_vector,
                         const ArrayDouble &previous_delta_dual,
                         const double l_l2sq) override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  void compute_lip_consts() override;

  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
    ar(cereal::make_nvp("ModelLipschitz", cereal::base_class<ModelLipschitz>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLinReg, cereal::specialization::member_serialize)

#endif  // TICK_OPTIM_MODEL_SRC_LINREG_H_
