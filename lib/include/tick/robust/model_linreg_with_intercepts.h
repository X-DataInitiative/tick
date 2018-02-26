#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_

// License: BSD 3 clause

#include "tick/linear_model/model_linreg.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"

template <class T>
class DLL_PUBLIC TModelLinRegWithIntercepts
    : virtual public TModelGeneralizedLinearWithIntercepts<T>,
      virtual public TModelLinReg<T> {
 protected:
  using TModelGeneralizedLinearWithIntercepts<T>::n_samples;
  using TModelGeneralizedLinearWithIntercepts<T>::n_features;
  using TModelGeneralizedLinearWithIntercepts<T>::compute_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::grad_i;
  using TModelGeneralizedLinearWithIntercepts<T>::fit_intercept;
  using TModelGeneralizedLinearWithIntercepts<T>::compute_grad_i;
  using TModelGeneralizedLinearWithIntercepts<T>::features;
  using TModelGeneralizedLinearWithIntercepts<T>::grad_i_factor;
  using TModelGeneralizedLinearWithIntercepts<T>::get_features_norm_sq;
  using TModelGeneralizedLinearWithIntercepts<T>::use_intercept;
  using TModelGeneralizedLinearWithIntercepts<T>::get_n_samples;
  using TModelLinReg<T>::ready_lip_consts;
  using TModelLinReg<T>::lip_consts;

 public:
  using TModelGeneralizedLinear<T>::get_class_name;

 public:
  TModelLinRegWithIntercepts(const std::shared_ptr<BaseArray2d<T> > features,
                             const std::shared_ptr<SArray<T> > labels,
                             const bool fit_intercept, const int n_threads = 1);

  virtual ~TModelLinRegWithIntercepts() {}

  void compute_lip_consts() override;
};

using ModelLinRegWithIntercepts = TModelLinRegWithIntercepts<double>;
using ModelLinRegWithInterceptsDouble = TModelLinRegWithIntercepts<double>;
using ModelLinRegWithInterceptsFloat = TModelLinRegWithIntercepts<float>;

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_LINREG_WITH_INTERCEPTS_H_
