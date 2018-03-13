#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class DLL_PUBLIC TProxL2 : public TProx<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProx<T>::strength;
  using TProx<T>::positive;

 public:
  using TProx<T>::get_class_name;

 protected:
  // This exists soley for cereal which has friend access
  TProxL2() : TProxL2<T>(0, 0, 1, false) {}

 public:
  TProxL2(T strength, bool positive);

  TProxL2(T strength, ulong start, ulong end, bool positive);

  T value(const Array<T>& coeffs, ulong start, ulong end) override;

  void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T> >(this)));
  }

  BoolStrReport compare(const TProxL2<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL2<T>& that) { return compare(that); }
};

using ProxL2 = TProxL2<double>;

using ProxL2Double = TProxL2<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2Double,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2Double)

using ProxL2Float = TProxL2<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2Float,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2Float)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2_H_
