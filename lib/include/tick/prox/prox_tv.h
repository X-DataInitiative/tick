#ifndef LIB_INCLUDE_TICK_PROX_PROX_TV_H_
#define LIB_INCLUDE_TICK_PROX_PROX_TV_H_

// License: BSD 3 clause

#include "prox.h"

template <class T, class K = T>
class DLL_PUBLIC TProxTV : public TProx<T, K> {
 protected:
  using TProx<T, K>::strength;
  using TProx<T, K>::positive;

 public:
  using TProx<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TProxTV() : TProxTV<T, K>(0, 0, 1, false) {}

  TProxTV(T strength, bool positive) : TProx<T, K>(strength, positive) {}

  TProxTV(T strength, ulong start, ulong end, bool positive)
      : TProx<T, K>(strength, start, end, positive) {}

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxTV<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxTV<T, K>& that) { return compare(that); }
};

using ProxTVDouble = TProxTV<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxTVDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxTVDouble)

using ProxTVFloat = TProxTV<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxTVFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxTVFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_TV_H_
