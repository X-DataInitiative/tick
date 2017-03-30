//
// Created by Stéphane GAIFFAS on 12/12/2015.
//

#ifndef TICK_OPTIM_MODEL_SRC_LINREG_WITH_INTERCEPTS_H_
#define TICK_OPTIM_MODEL_SRC_LINREG_WITH_INTERCEPTS_H_

#include "model_generalized_linear_with_intercepts.h"
#include "model_lipschitz.h"

class ModelLinRegWithIntercepts : public ModelGeneralizedLinearWithIntercepts,
                                  public ModelLipschitz {
 public:
  ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                            const SArrayDoublePtr labels,
                            const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  void compute_lip_consts() override;
};

#endif  // TICK_OPTIM_MODEL_SRC_LINREG_WITH_INTERCEPTS_H_
