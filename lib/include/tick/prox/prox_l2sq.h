#ifndef LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_

// License: BSD 3 clause

#include "prox_separable.h"

template <class T, class K = T>
class DLL_PUBLIC TProxL2Sq : public TProxSeparable<T, K> {
 protected:
  using TProxSeparable<T, K>::has_range;
  using TProxSeparable<T, K>::strength;
  using TProxSeparable<T, K>::start;
  using TProxSeparable<T, K>::end;
  using TProxSeparable<T, K>::positive;

 public:
  using TProxSeparable<T, K>::get_class_name;

 public:
  // This exists soley for cereal/swig
  TProxL2Sq() : TProxL2Sq<T, K>(0, 0, 1, false) {}

  TProxL2Sq(T strength, bool positive)
      : TProxSeparable<T, K>(strength, positive) {}

  TProxL2Sq(T strength, ulong start, ulong end, bool positive)
      : TProxSeparable<T, K>(strength, start, end, positive) {}

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K> >(this)));
  }

  BoolStrReport compare(const TProxL2Sq<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL2Sq<T, K>& that) {
    return compare(that);
  }

 protected:
  T value_single(T x) const override;

  T call_single(T x, T step) const override;

  // Repeat n_times the prox on coordinate i
  T call_single(T x, T step, ulong n_times) const override;
};

using ProxL2Sq = TProxL2Sq<double, double>;

using ProxL2SqDouble = TProxL2Sq<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2SqDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2SqDouble)

using ProxL2SqFloat = TProxL2Sq<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL2SqFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL2SqFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L2SQ_H_
