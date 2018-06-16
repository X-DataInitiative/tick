#ifndef LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TProxElasticNet : public TProxSeparable<T, K> {
 protected:
  using TProxSeparable<T, K>::strength;
  using TProxSeparable<T, K>::positive;

 public:
  using TProxSeparable<T, K>::get_class_name;

 protected:
  T ratio = 0;

 public:
  // This exists soley for cereal/swig
  TProxElasticNet() : TProxElasticNet<T, K>(0, 0, false) {}

  TProxElasticNet(T strength, T ratio, bool positive)
      : TProxSeparable<T, K>(strength, positive) {
    this->positive = positive;
    set_ratio(ratio);
  }

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive)
      : TProxSeparable<T, K>(strength, start, end, positive) {
    this->positive = positive;
    set_ratio(ratio);
  }

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K> >(this)));

    ar(CEREAL_NVP(ratio));
  }

  BoolStrReport compare(const TProxElasticNet<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal =
        TProxSeparable<T, K>::compare(that, ss) && TICK_CMP_REPORT(ss, ratio);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxElasticNet<T, K>& that) {
    return compare(that);
  }

 private:
  T call_single(T x, T step) const override;

  T value_single(T x) const override;
};

using ProxElasticNet = TProxElasticNet<double, double>;

using ProxElasticNetDouble = TProxElasticNet<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetDouble)

using ProxElasticNetFloat = TProxElasticNet<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetFloat)

using ProxElasticNetAtomicDouble =
    TProxElasticNet<double, std::atomic<double> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetAtomicDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetAtomicDouble)

using ProxElasticNetAtomicFloat = TProxElasticNet<float, std::atomic<float> >;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetAtomicFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetAtomicFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
