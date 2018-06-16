#ifndef LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_

// License: BSD 3 clause

#include "prox.h"
#include "prox_sorted_l1.h"

template <class T, class K = T>
class DLL_PUBLIC TProxSlope : public TProxSortedL1<T, K> {
 protected:
  using TProxSortedL1<T, K>::start;
  using TProxSortedL1<T, K>::end;
  using TProxSortedL1<T, K>::weights_ready;
  using TProxSortedL1<T, K>::weights;
  using TProxSortedL1<T, K>::strength;

 public:
  using TProxSortedL1<T, K>::get_class_name;

 protected:
  T false_discovery_rate;
  void compute_weights(void) override;

 public:
  // This exists soley for cereal/swig
  TProxSlope() : TProxSlope(0, 0, 0, 1, false) {}

  TProxSlope(T strength, T false_discovery_rate, bool positive)
      : TProxSortedL1<T, K>(strength, WeightsType::bh, positive) {
    this->false_discovery_rate = false_discovery_rate;
  }

  TProxSlope(T strength, T false_discovery_rate, ulong start, ulong end,
             bool positive)
      : TProxSortedL1<T, K>(strength, WeightsType::bh, start, end, positive) {
    this->false_discovery_rate = false_discovery_rate;
  }

  inline T get_false_discovery_rate() const { return false_discovery_rate; }

  inline void set_false_discovery_rate(T false_discovery_rate) {
    if (false_discovery_rate <= 0 || false_discovery_rate >= 1) {
      TICK_ERROR("False discovery rate must be in (0, 1) but received "
                 << false_discovery_rate)
    }
    if (false_discovery_rate != this->false_discovery_rate) {
      weights_ready = false;
    }
    this->false_discovery_rate = false_discovery_rate;
  }

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProx<T, K> >(this)));

    ar(CEREAL_NVP(false_discovery_rate));
  }

  BoolStrReport compare(const TProxSlope<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxSlope<T, K>& that) {
    return compare(that);
  }
};

using ProxSlopeDouble = TProxSlope<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSlopeDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxSlopeDouble)

using ProxSlopeFloat = TProxSlope<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSlopeFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxSlopeFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SLOPE_H_
