#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T>
class DLL_PUBLIC TProxL1 : public TProxSeparable<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProx<T>::has_range;
  using TProx<T>::strength;
  using TProx<T>::start;
  using TProx<T>::end;
  using TProx<T>::positive;

 public:
  using TProxSeparable<T>::get_class_name;

 protected:
  // This exists soley for cereal which has friend access
  TProxL1() : TProxL1<T>(0, 0) {}

  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;

  T value_single(T x) const override;

 public:
  TProxL1(T strength, bool positive);

  TProxL1(T strength, ulong start, ulong end, bool positive);

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T> >(this)));
  }

  BoolStrReport compare(const TProxL1<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL1<T>& that) { return compare(that); }
};

using ProxL1 = TProxL1<double>;

using ProxL1Double = TProxL1<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1Double,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1Double)

using ProxL1Float = TProxL1<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1Float,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1Float)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1_H_
