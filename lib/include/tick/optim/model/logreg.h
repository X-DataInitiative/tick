//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_LOGREG_H_
#define TICK_OPTIM_MODEL_SRC_LOGREG_H_

// License: BSD 3 clause

#include "model_generalized_linear.h"
#include "model_lipschitz.h"


// TODO: labels should be a ArrayInt

class DLL_PUBLIC ModelLogReg : public ModelGeneralizedLinear, public ModelLipschitz {
 public:
  // Empty constructor only used for serialization
  ModelLogReg(){};

  ModelLogReg(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads = 1);

  const char *get_class_name() const override;

  static inline double sigmoid(const double z) {
    // Overflow-proof sigmoid
    if (z > 0) {
      return 1 / (1 + exp(-z));
    } else {
      const double exp_z = exp(z);
      return exp_z / (1 + exp_z);
    }
  }

  static inline double logistic(const double z) {
    if (z > 0) {
      return log(1 + exp(-z));
    } else {
      return -z + log(1 + exp(z));
    }
  }

  static void sigmoid(const ArrayDouble &x, ArrayDouble &out);

  static void logistic(const ArrayDouble &x, ArrayDouble &out);

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  double sdca_dual_min_i(const ulong i,
                         const double dual_i,
                         const ArrayDouble &primal_vector,
                         const double previous_delta_dual_i,
                         double l_l2sq) override;

  void compute_lip_consts() override;

 public:
  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
    ar(cereal::make_nvp("ModelLipschitz", cereal::base_class<ModelLipschitz>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLogReg, cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLogReg)

#endif  // TICK_OPTIM_MODEL_SRC_LOGREG_H_
