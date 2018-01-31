//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

// TODO: labels should be a ArrayInt (PD: should they?)
template <class T, class K = T>
class DLL_PUBLIC TModelLogReg : public TModelGeneralizedLinear<T, K>, virtual public TModelLipschitz<T, K> {
 private:
  std::string clazz = "TModelLabelsFeatures<"
    + std::string(typeid(T).name())
    + ", " + std::string(typeid(K).name()) + ">";

 protected:
  using TModelLipschitz<T, K>::ready_lip_consts;
  using TModelLipschitz<T, K>::lip_consts;

  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::get_label;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::use_intercept;
  using TModelGeneralizedLinear<T, K>::get_inner_prod;
  using TModelGeneralizedLinear<T, K>::fit_intercept;

 public:
  TModelLogReg(
    const std::shared_ptr<BaseArray2d<K> > features,
    const std::shared_ptr<SArray<K> > labels,
    const bool fit_intercept,
    const int n_threads = 1);

  virtual const char *get_class_name() const {
    return clazz.c_str();
  }

  static inline K sigmoid(const K z) {
    // Overflow-proof sigmoid
    if (z > 0) {
      return 1 / (1 + exp(-z));
    } else {
      const K exp_z = exp(z);
      return exp_z / (1 + exp_z);
    }
  }

  static inline K logistic(const K z) {
    if (z > 0) {
      return log(1 + exp(-z));
    } else {
      return -z + log(1 + exp(z));
    }
  }

  static void sigmoid(const Array<K> &x, Array<K> &out);

  static void logistic(const Array<K> &x, Array<K> &out);

  K loss_i(const ulong i, const Array<T> &coeffs) override;

  K grad_i_factor(const ulong i, const Array<T> &coeffs) override;

  K sdca_dual_min_i(const ulong i,
                         const K dual_i,
                         const Array<K> &primal_vector,
                         const K previous_delta_dual_i,
                         K l_l2sq) override;

  void compute_lip_consts() override;
};


class DLL_PUBLIC ModelLogReg : public TModelLogReg<double, double>{
 public:
  using TModelLogReg<double, double>::sigmoid;
  using TModelLogReg<double, double>::logistic;
 public:
  ModelLogReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads = 1);

  const char *get_class_name() const override;

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);

  static void logistic(const ArrayDouble &x, ArrayDouble &out);
};

using ModelLogRegDouble = TModelLogReg<double, double>;
using ModelLogRegFloat  = TModelLogReg<float , float>;

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
