//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_POISREG_H_
#define TICK_OPTIM_MODEL_SRC_POISREG_H_

#include "model_generalized_linear.h"


// TODO: labels should be a ArrayUInt

enum class LinkType {
  identity = 0,
  exponential
};

class ModelPoisReg : public ModelGeneralizedLinear {
 private:
  LinkType link_type;

 public:
  ModelPoisReg(const SBaseArrayDouble2dPtr features,
               const SArrayDoublePtr labels,
               const LinkType link_type,
               const bool fit_intercept,
               const int n_threads = 1);

  const char *get_class_name() const override {
    return "ModelPoisReg";
  }

  double sdca_dual_min_i(const ulong i,
                         const ArrayDouble &dual_vector,
                         const ArrayDouble &primal_vector,
                         const ArrayDouble &previous_delta_dual,
                         const double l_l2sq) override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  virtual void set_link_type(const LinkType link_type) {
    this->link_type = link_type;
  }
};

#endif  // TICK_OPTIM_MODEL_SRC_POISREG_H_
