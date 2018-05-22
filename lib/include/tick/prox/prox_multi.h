
#ifndef LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
#define LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_

// License: BSD 3 clause

#include "prox.h"

// TODO: this requires some work. ProxMulti should have the standard
// TODO: prox API, with a set_strength, and things like that

template <class T, class K = T>
class DLL_PUBLIC TProxMulti : public TProx<T, K> {
 protected:
  using ProxTPtrVector = std::vector<std::shared_ptr<TProx<T, K> > >;

 public:
  using TProx<T, K>::get_class_name;

 protected:
  ProxTPtrVector proxs;

 protected:
  // This exists soley for cereal/swig
  TProxMulti() {}

 public:
  explicit TProxMulti(ProxTPtrVector proxs);

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  void call(const Array<K>& coeffs, T step, Array<K>& out, ulong start,
            ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProx<T, K> >(this)));
    ar(CEREAL_NVP(proxs));
  }

  BoolStrReport compare(const TProxMulti<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxMulti<T, K>& that) {
    return compare(that);
  }
};

using ProxMultiDouble = TProxMulti<double, double>;
using ProxMultiFloat = TProxMulti<float, float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
