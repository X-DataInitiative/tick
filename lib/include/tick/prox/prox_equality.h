#ifndef LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_

// License: BSD 3 clause

#include "prox.h"

template <class T, class K = T>
class DLL_PUBLIC TProxEquality : public TProx<T, K> {
 protected:
  using TProx<T, K>::positive;

 public:
  using TProx<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TProxEquality() : TProxEquality<T, K>(0, false) {}

  explicit TProxEquality(T strength, bool positive)
      : TProx<T, K>(0., positive) {}

  TProxEquality(T strength, ulong start, ulong end, bool positive)
      : TProx<T, K>(0., start, end, positive) {}

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxEquality<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxEquality<T, K>& that) {
    return compare(that);
  }
};

using ProxEqualityDouble = TProxEquality<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxEqualityDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxEqualityDouble)

using ProxEqualityFloat = TProxEquality<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxEqualityFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxEqualityFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
