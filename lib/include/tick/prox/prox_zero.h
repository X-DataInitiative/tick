
#ifndef LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TProxZero : public TProxSeparable<T, K> {
 public:
  using TProxSeparable<T, K>::get_class_name;
  using TProxSeparable<T, K>::value;

 private:
  T call_single(T x, T step) const override;

  T call_single(T x, T step, ulong n_times) const override;

 public:
  // This exists soley for cereal/swig
  TProxZero() : TProxZero<T, K>(0) {}

  explicit TProxZero(T strength) : TProxSeparable<T, K>(strength, false) {}

  TProxZero(T strength, ulong start, ulong end)
      : TProxSeparable<T, K>(strength, start, end, false) {}

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K> >(this)));
  }

  BoolStrReport compare(const TProxZero<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxZero<T, K>& that) {
    return compare(that);
  }
};

using ProxZeroDouble = TProxZero<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxZeroDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxZeroDouble)

using ProxZeroFloat = TProxZero<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxZeroFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxZeroFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
