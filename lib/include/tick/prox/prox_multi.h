
#ifndef LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
#define LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_

// License: BSD 3 clause

#include "prox.h"

// TODO: this requires some work. ProxMulti should have the standard
// TODO: prox API, with a set_strength, and things like that

template <class T>
class DLL_PUBLIC TProxMulti : public TProx<T> {
  // Grants cereal access to default constructor
  friend class cereal::access;

 protected:
  using ProxTPtrVector = std::vector<std::shared_ptr<TProx<T> > >;

 public:
  using TProx<T>::get_class_name;

 protected:
  ProxTPtrVector proxs;

 protected:
  // This exists soley for cereal which has friend access
  TProxMulti() {}

 public:
  explicit TProxMulti(ProxTPtrVector proxs);

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void call(const Array<T> &coeffs, T step, Array<T> &out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProx<T> >(this)));
    ar(CEREAL_NVP(proxs));
  }

  BoolStrReport compare(const TProxMulti<T>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxMulti<T>& that) { return compare(that); }
};

using ProxMulti = TProxMulti<double>;
using ProxMultiDouble = TProxMulti<double>;
using ProxMultiFloat = TProxMulti<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
