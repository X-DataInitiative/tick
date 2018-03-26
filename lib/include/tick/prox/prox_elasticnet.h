#ifndef LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxElasticNet : public TProxSeparable<T> {
 protected:
  using TProxSeparable<T>::strength;
  using TProxSeparable<T>::positive;

 public:
  using TProxSeparable<T>::get_class_name;

 protected:
  T ratio = 0;

 public:
  // This exists soley for cereal/swig
  TProxElasticNet() : TProxElasticNet<T>(0, 0, false) {}

  TProxElasticNet(T strength, T ratio, bool positive)
      : TProxSeparable<T>(strength, positive) {
    this->positive = positive;
    set_ratio(ratio);
  }

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive)
      : TProxSeparable<T>(strength, start, end, positive) {
    this->positive = positive;
    set_ratio(ratio);
  }

  virtual T get_ratio() const;

  virtual void set_ratio(T ratio);

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T> >(this)));

    ar(CEREAL_NVP(ratio));
  }

  BoolStrReport compare(const TProxElasticNet<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T>::compare(that, ss) && TICK_CMP_REPORT(ss, ratio);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxElasticNet<T>& that) {
    return compare(that);
  }

 private:
  T call_single(T x, T step) const override;

  T value_single(T x) const override;
};

using ProxElasticNet = TProxElasticNet<double>;
using ProxElasticNetDouble = TProxElasticNet<double>;
using ProxElasticNetFloat = TProxElasticNet<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetDouble)


CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
