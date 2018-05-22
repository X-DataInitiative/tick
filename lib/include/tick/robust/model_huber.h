
#ifndef LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_
#define LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_

// License: BSD 3 clause

#include "tick/base_model/model_generalized_linear.h"
#include "tick/base_model/model_lipschitz.h"

template <class T, class K = T>
class DLL_PUBLIC TModelHuber : public virtual TModelGeneralizedLinear<T, K>,
                               public TModelLipschitz<T, K> {
 protected:
  using TModelGeneralizedLinear<T, K>::features_norm_sq;
  using TModelGeneralizedLinear<T, K>::compute_features_norm_sq;
  using TModelGeneralizedLinear<T, K>::n_samples;
  using TModelGeneralizedLinear<T, K>::n_features;
  using TModelGeneralizedLinear<T, K>::fit_intercept;
  using TModelGeneralizedLinear<T, K>::compute_grad_i;
  using TModelGeneralizedLinear<T, K>::n_threads;
  using TModelGeneralizedLinear<T, K>::get_inner_prod;
  using TModelLipschitz<T, K>::ready_lip_consts;
  using TModelLipschitz<T, K>::lip_consts;

 public:
  using TModelGeneralizedLinear<T, K>::get_label;
  using TModelGeneralizedLinear<T, K>::grad_i;
  using TModelGeneralizedLinear<T, K>::get_features;
  using TModelGeneralizedLinear<T, K>::grad_i_factor;
  using TModelGeneralizedLinear<T, K>::get_class_name;

 private:
  T threshold, threshold_squared_over_two;

 public:
  // This exists soley for cereal/swig
  TModelHuber() : TModelHuber(nullptr, nullptr, false, 1) {}

  TModelHuber(const std::shared_ptr<BaseArray2d<T> > features,
              const std::shared_ptr<SArray<T> > labels,
              const bool fit_intercept, const T threshold,
              const int n_threads = 1)
      : TModelLabelsFeatures<T, K>(features, labels),
        TModelGeneralizedLinear<T, K>(features, labels, fit_intercept,
                                      n_threads) {
    set_threshold(threshold);
  }

  T loss_i(const ulong i, const Array<K> &coeffs) override;

  T grad_i_factor(const ulong i, const Array<K> &coeffs) override;

  void compute_lip_consts() override;

  virtual T get_threshold(void) const { return threshold; }

  virtual void set_threshold(const T threshold) {
    if (threshold <= 0.) {
      TICK_ERROR("threshold must be > 0");
    } else {
      this->threshold = threshold;
      threshold_squared_over_two = threshold * threshold / 2;
    }
  }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp(
        "ModelGeneralizedLinear",
        cereal::base_class<TModelGeneralizedLinear<T, K> >(this)));
    ar(cereal::make_nvp("ModelLipschitz",
                        cereal::base_class<TModelLipschitz<T, K> >(this)));
    ar(threshold);
    ar(threshold_squared_over_two);
  }

  BoolStrReport compare(const TModelHuber<T, K> &that, std::stringstream &ss) {
    ss << get_class_name() << std::endl;
    bool are_equal = TModelGeneralizedLinear<T, K>::compare(that, ss) &&
                     TModelLipschitz<T, K>::compare(that, ss) &&
                     TICK_CMP_REPORT(ss, threshold) &&
                     TICK_CMP_REPORT(ss, threshold_squared_over_two);

    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport compare(const TModelHuber<T, K> &that) {
    std::stringstream ss;
    return compare(that, ss);
  }
  BoolStrReport operator==(const TModelHuber<T, K> &that) {
    return TModelHuber<T, K>::compare(that);
  }
};

using ModelHuber = TModelHuber<double>;
using ModelHuberDouble = TModelHuber<double>;
using ModelHuberFloat = TModelHuber<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHuberDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHuberDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelHuberFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ModelHuberFloat)

#endif  // LIB_INCLUDE_TICK_ROBUST_MODEL_HUBER_H_
