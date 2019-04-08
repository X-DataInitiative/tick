#ifndef LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_

// License: BSD 3 clause

#include "prox.h"

enum class WeightsType : uint16_t { bh = 0, oscar };

template <class T, class K = T>
class DLL_PUBLIC TProxSortedL1 : public TProx<T, K> {
 protected:
  using TProx<T, K>::start;
  using TProx<T, K>::end;
  using TProx<T, K>::strength;

 public:
  using TProx<T, K>::get_class_name;

 protected:
  WeightsType weights_type;
  Array<T> weights;
  bool weights_ready;

  virtual void compute_weights(void);

  void prox_sorted_l1(const Array<T>& y, const Array<T>& strength,
                      Array<T>& x) const;

 public:
  // This exists soley for cereal/swig
  TProxSortedL1() : TProxSortedL1(0, WeightsType::bh, 0, 1, false) {}

  TProxSortedL1(T strength, WeightsType weights_type, bool positive)
      : TProx<T, K>(strength, positive) {
    this->weights_type = weights_type;
    weights_ready = false;
  }

  TProxSortedL1(T strength, WeightsType weights_type, ulong start, ulong end,
                bool positive)
      : TProx<T, K>(strength, start, end, positive) {
    this->weights_type = weights_type;
    weights_ready = false;
  }

  T value(const Array<K>& coeffs, ulong start, ulong end) override;

  void call(const Array<K>& coeffs, T t, Array<K>& out, ulong start,
            ulong end) override;

  inline WeightsType get_weights_type() const { return weights_type; }

  inline void set_weights_type(WeightsType weights_type) {
    this->weights_type = weights_type;
    weights_ready = false;
  }

  inline T get_weight_i(ulong i) { return weights[i]; }

  void set_strength(T strength) override;

  void set_start_end(ulong start, ulong end) override;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("Prox", cereal::base_class<TProx<T, K> >(this)));
  }

  BoolStrReport compare(const TProxSortedL1<T, K>& that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProx<T, K>::compare(that, ss);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxSortedL1<T, K>& that) {
    return compare(that);
  }
};

using ProxSortedL1Double = TProxSortedL1<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSortedL1Double,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxSortedL1Double)

using ProxSortedL1Float = TProxSortedL1<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxSortedL1Float,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxSortedL1Float)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
