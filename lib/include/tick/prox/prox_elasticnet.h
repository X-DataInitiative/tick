#ifndef LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxElasticNet : public TProxSeparable<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProxSeparable<T>::strength;
  using TProxSeparable<T>::positive;

 public:
  using TProxSeparable<T>::get_class_name;

 protected:
  T ratio = 0;

 protected:
  // This exists soley for cereal which has friend access
  TProxElasticNet() : TProxElasticNet<T>(0, 0, false) {}

 public:
  TProxElasticNet(T strength, T ratio, bool positive);

  TProxElasticNet(T strength, T ratio, ulong start, ulong end, bool positive);

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
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetDouble)

using ProxElasticNetFloat = TProxElasticNet<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxElasticNetFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxElasticNetFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ELASTICNET_H_
