
#ifndef LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
#define LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxZero : public TProxSeparable<T> {
 public:
  using TProxSeparable<T>::get_class_name;

 private:
  T call_single(T x, T step) const override;

  T call_single(T x, T step, ulong n_times) const override;

 public:
  // This exists soley for cereal/swig
  TProxZero() : TProxZero(0) {}

  explicit TProxZero(T strength) : TProxSeparable<T>(strength, false) {}

  TProxZero(T strength, ulong start, ulong end)
      : TProxSeparable<T>(strength, start, end, false) {}

  T value(const Array<T>& coeffs, ulong start, ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T> >(this)));
  }

  BoolStrReport compare(const TProxZero<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxZero<T>& that) { return compare(that); }
};

using ProxZero = TProxZero<double>;
using ProxZeroDouble = TProxZero<double>;
using ProxZeroFloat = TProxZero<float>;

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxZeroDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxZeroDouble)

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxZeroFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxZeroFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_ZERO_H_
