#ifndef LIB_INCLUDE_TICK_PROX_PROX_TV_H_
#define LIB_INCLUDE_TICK_PROX_PROX_TV_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class DLL_PUBLIC TProxTV : public TProx<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProx<T>::strength;
  using TProx<T>::positive;

 public:
  using TProx<T>::get_class_name;

 protected:
  // This exists soley for cereal which has friend access
  TProxTV() : TProxTV(0, 0, 1, false) {}

 public:
  TProxTV(T strength, bool positive);

  TProxTV(T strength, ulong start, ulong end, bool positive);

  T value(const Array<T>& coeffs, ulong start, ulong end) override;

  void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T> >(this)));
  }

  BoolStrReport compare(const TProxTV<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxTV<T>& that) { return compare(that); }
};

using ProxTV = TProxTV<double>;

using ProxTVDouble = TProxTV<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxTVDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxTVDouble)

using ProxTVFloat = TProxTV<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxTVFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxTVFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_TV_H_
