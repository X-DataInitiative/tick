#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_

// License: BSD 3 clause


#include "tick/robust/model_generalized_linear_with_intercepts.h"
#include "tick/linear_model/model_linreg.h"

class DLL_PUBLIC ModelLinRegWithIntercepts : public ModelGeneralizedLinearWithIntercepts,
                                  public ModelLinReg {
 public:
  ModelLinRegWithIntercepts(const SBaseArrayDouble2dPtr features,
                            const SArrayDoublePtr labels,
                            const bool fit_intercept,
                            const int n_threads = 1);

  const char *get_class_name() const override;

  void compute_lip_consts() override;
};

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
