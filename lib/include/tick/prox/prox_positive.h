#ifndef LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K>
class DLL_PUBLIC TProxPositive : public TProxSeparable<T, K> {
 public:
  using TProxSeparable<T, K>::get_class_name;

 private:
  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;

 public:
  // This exists soley for cereal/swig
  TProxPositive() : TProxPositive(0) {}

  explicit TProxPositive(T strength) : TProxSeparable<T, K>(strength, true) {}

  TProxPositive(T strength, ulong start, ulong end)
      : TProxSeparable<T, K>(strength, start, end, true) {}

  // Override value, only this value method should be called
  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K> >(this)));
  }

  BoolStrReport compare(const TProxPositive<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto super = TProxSeparable<T, K>::compare(that, ss);
    return BoolStrReport(super, ss.str());
  }
  BoolStrReport operator==(const TProxPositive<T, K>& that) {
    return compare(that);
  }
};

using ProxPositiveDouble = TProxPositive<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxPositiveDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxPositiveDouble)

using ProxPositiveFloat = TProxPositive<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxPositiveFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxPositiveFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_POSITIVE_H_
