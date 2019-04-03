//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
#define LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T, class K = T>
class DLL_PUBLIC TModelLogReg : public virtual TModelGeneralizedLinear<T, K>,
                                public TModelLipschitz<T, K> {
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
  using TModelGeneralizedLinear<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TModelLogReg() : TModelLogReg<T, K>(nullptr, nullptr, 0, 0) {}

  TModelLogReg(const std::shared_ptr<BaseArray2d<T> > features,
               const std::shared_ptr<SArray<T> > labels,
               const bool fit_intercept, const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {}

  virtual ~TModelLogReg() {}

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

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  T sdca_dual_min_i(const ulong i, const T dual_i,
                    const T primal_dot_features,
                    const T previous_delta_dual_i, T _1_over_lbda_n) override;

  Array<T> sdca_dual_min_many(ulong indices,
                              const Array<T> &duals,
                              Array2d<T> &g,
                              Array2d<T> &n_hess,
                              Array<T> &p,
                              Array<T> &n_grad,
                              Array<T> &sdca_labels,
                              Array<T> &new_duals,
                              Array<T> &delta_duals,
                              ArrayInt &ipiv) override;

  void compute_lip_consts() override;

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        typename cereal::virtual_base_class<TModelGeneralizedLinear<T, K> >(
            this)));
    ar(cereal::make_nvp(
        "ModelLipschitz",
        typename cereal::base_class<TModelLipschitz<T, K> >(this)));
  }

  BoolStrReport compare(const TModelLogReg<T, K> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TModelLipschitz<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelLogReg<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelLogReg<T, K> &that) {
    return TModelLogReg<T, K>::compare(that);
  }
};

using ModelLogReg = TModelLogReg<double, double>;

using ModelLogRegDouble = TModelLogReg<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLogRegDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLogRegDouble)

using ModelLogRegFloat = TModelLogReg<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLogRegFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLogRegFloat)

using ModelLogRegAtomic = TModelLogReg<double, std::atomic<double> >;
using ModelLogRegAtomicDouble = TModelLogReg<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLogRegAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLogRegAtomicDouble)

using ModelLogRegAtomicFloat = TModelLogReg<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelLogRegAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelLogRegAtomicFloat)

#endif  // LIB_INCLUDE_TICK_LINEAR_MODEL_MODEL_LOGREG_H_
