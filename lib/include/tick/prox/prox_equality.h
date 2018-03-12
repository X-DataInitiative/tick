#ifndef LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_

// License: BSD 3 clause

#include "prox.h"

template <class T>
class DLL_PUBLIC TProxEquality : public TProx<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using TProx<T>::positive;

 public:
  using TProx<T>::get_class_name;

 private:
  // This exists soley for cereal which has friend access
  TProxEquality() : TProxEquality<T>(0, false) {}

 public:
  explicit TProxEquality(T strength, bool positive);

  TProxEquality(T strength, ulong start, ulong end, bool positive);

  T value(const Array<T>& coeffs, ulong start, ulong end) override;

  void call(const Array<T>& coeffs, T step, Array<T>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable", cereal::base_class<TProx<T> >(this)));
  }

  BoolStrReport compare(const TProxEquality<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxEquality<T>& that) {
    return compare(that);
  }
};

using ProxEquality = TProxEquality<double>;

using ProxEqualityDouble = TProxEquality<double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxEqualityDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxEqualityDouble)

using ProxEqualityFloat = TProxEquality<float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxEqualityFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxEqualityFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_EQUALITY_H_
