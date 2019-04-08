#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TProxL1 : public TProxSeparable<T, K> {
 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::strength;
  using TProx<T, K>::start;
  using TProx<T, K>::end;
  using TProx<T, K>::positive;

 public:
  using TProxSeparable<T, K>::get_class_name;

 protected:
  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;

  T value_single(T x) const override;

 public:
  // This exists soley for cereal/swig
  TProxL1() : TProxL1<T, K>(0, 0) {}

  TProxL1(T strength, bool positive)
      : TProxSeparable<T, K>(strength, positive) {}

  TProxL1(T strength, ulong start, ulong end, bool positive)
      : TProxSeparable<T, K>(strength, start, end, positive) {}

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K> >(this)));
  }

  BoolStrReport compare(const TProxL1<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL1<T, K>& that) { return compare(that); }
};

using ProxL1Double = TProxL1<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1Double,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1Double)

using ProxL1Float = TProxL1<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1Float,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1Float)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1_H_
