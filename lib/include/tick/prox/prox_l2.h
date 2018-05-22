#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2_H_

// License: BSD 3 clause

#include "prox.h"

template <class T, class K>
class DLL_PUBLIC TProxL2 : public TProx<T, K> {
 protected:
  using TProx<T, K>::strength;
  using TProx<T, K>::positive;

 public:
  using TProx<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TProxL2() : TProxL2<T, K>(0, 0, 1, false) {}

  TProxL2(T strength, bool positive) : TProx<T, K>(strength, positive) {}

  TProxL2(T strength, ulong start, ulong end, bool positive)
      : TProx<T, K>(strength, start, end, positive) {}

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxL2<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL2<T, K>& that) { return compare(that); }
};

using ProxL2Double = TProxL2<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2Double,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2Double)

using ProxL2Float = TProxL2<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2Float,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2Float)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2_H_
