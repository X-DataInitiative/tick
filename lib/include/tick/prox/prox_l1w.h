#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1W_H_

// License: BSD 3 clause

#include "prox_separable.h"
#include "tick/base/base.h"

template <class T, class K = T>
class DLL_PUBLIC TProxL1w : public TProxSeparable<T, K> {
 protected:
  using TProx<T, K>::has_range;
  using TProx<T, K>::positive;
  using TProx<T, K>::strength;
  using TProx<T, K>::start;
  using TProx<T, K>::end;

 public:
  using TProxSeparable<T, K>::get_class_name;

 public:
  using SArrayTPtr = std::shared_ptr<SArray<T>>;

 protected:
  // Weights for L1 penalization
  SArrayTPtr weights;

 public:
  // This exists soley for cereal/swig
  TProxL1w() : TProxL1w<T, K>(0, nullptr, 0) {}

  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, bool positive)
      : TProxSeparable<T, K>(strength, positive) {
    this->weights = weights;
  }

  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, ulong start,
           ulong end, bool positive)
      : TProxSeparable<T, K>(strength, start, end, positive) {
    this->weights = weights;
  }

  void call(const Array<K> &coeffs, const T step, Array<K> &out, ulong start,
            ulong end) override;

  void call(const Array<K> &coeffs, const Array<T> &step, Array<K> &out,
            ulong start, ulong end) override;

  // For this prox we cannot only override T call_single(T, step) const,
  // since we need the weights...
  void call_single(ulong i, const Array<K> &coeffs, T step,
                   Array<K> &out) const override;

  void call_single(ulong i, const Array<K> &coeffs, T step, Array<K> &out,
                   ulong n_times) const override;

  T value(const Array<K> &coeffs, ulong start, ulong end) override;

  void set_weights(SArrayTPtr weights) { this->weights = weights; }

  template <class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ProxSeparable",
                        cereal::base_class<TProxSeparable<T, K>>(this)));

    ar(CEREAL_NVP(weights));
  }

  BoolStrReport compare(const TProxL1w<T, K> &that) {
    std::stringstream ss;
    ss << get_class_name();
    auto are_equal = TProxSeparable<T, K>::compare(that, ss) &&
                     TICK_CMP_REPORT_PTR(ss, weights);
    return BoolStrReport(are_equal, ss.str());
  }
  BoolStrReport operator==(const TProxL1w<T, K> &that) { return compare(that); }

 private:
  T call_single(T x, T step) const override;

  T call_single(T x, T step, ulong n_times) const override;

  T value_single(T x) const override;

  T call_single(T x, T step, T weight) const;

  T call_single(T x, T step, T weight, ulong n_times) const;

  T value_single(T x, T weight) const;
};

using ProxL1wDouble = TProxL1w<double, double>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1wDouble,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1wDouble)

using ProxL1wFloat = TProxL1w<float, float>;
CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ProxL1wFloat,
                                   cereal::specialization::member_serialize)
CEREAL_REGISTER_TYPE(ProxL1wFloat)

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
