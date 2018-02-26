//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

// TODO: labels should be a ArrayInt (PD: should they?)
template <class T>
class DLL_PUBLIC TModelLogReg : public TModelGeneralizedLinear<T>,
                                virtual public TModelLipschitz<T> {
 private:
  std::string clazz =
      "TModelLabelsFeatures<" + std::string(typeid(T).name()) + ">";

 protected:
  using TModelLipschitz<T>::ready_lip_consts;
  using TModelLipschitz<T>::lip_consts;

  using TModelGeneralizedLinear<T>::n_samples;
  using TModelGeneralizedLinear<T>::get_label;
  using TModelGeneralizedLinear<T>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T>::features_norm_sq;
  using TModelGeneralizedLinear<T>::use_intercept;
  using TModelGeneralizedLinear<T>::get_inner_prod;
  using TModelGeneralizedLinear<T>::fit_intercept;
  using TModelGeneralizedLinear<T>::get_class_name;

 public:
  TModelLogReg(const std::shared_ptr<BaseArray2d<T> > features,
               const std::shared_ptr<SArray<T> > labels,
               const bool fit_intercept, const int n_threads = 1);

  static inline T sigmoid(const T z) {
    // Overflow-proof sigmoid
    if (z > 0) {
      return 1 / (1 + exp(-z));
    } else {
      const T exp_z = exp(z);
      return exp_z / (1 + exp_z);
    }
  }

  static inline T logistic(const T z) {
    if (z > 0) {
      return log(1 + exp(-z));
    } else {
      return -z + log(1 + exp(z));
    }
  }

  static void sigmoid(const Array<T> &x, Array<T> &out);

  static void logistic(const Array<T> &x, Array<T> &out);

  T loss_i(const ulong i, const Array<T> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  T sdca_dual_min_i(const ulong i, const T dual_i,
                    const Array<T> &primal_vector,
                    const T previous_delta_dual_i, T l_l2sq) override;

  void compute_lip_consts() override;
};

using ModelLogReg = TModelLogReg<double>;
using ModelLogRegDouble = TModelLogReg<double>;
using ModelLogRegFloat = TModelLogReg<float>;

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
