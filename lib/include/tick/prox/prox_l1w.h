#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1W_H_

// License: BSD 3 clause

#include "prox_separable.h"
#include "tick/base/base.h"

template <class T>
class DLL_PUBLIC TProxL1w : public TProxSeparable<T> {
 protected:
  using TProxSeparable<T>::has_range;
  using TProxSeparable<T>::positive;
  using TProxSeparable<T>::strength;
  using TProxSeparable<T>::start;
  using TProxSeparable<T>::end;

 public:
  using TProxSeparable<T>::get_class_name;

 public:
  using SArrayTPtr = std::shared_ptr<SArray<T>>;

 protected:
  // Weights for L1 penalization
  SArrayTPtr weights;

 public:
  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, bool positive);

  TProxL1w(T strength, std::shared_ptr<SArray<T>> weights, ulong start,
           ulong end, bool positive);

  void call(const Array<T> &coeffs, const T step, Array<T> &out, ulong start,
            ulong end) override;

  void call(const Array<T> &coeffs, const Array<T> &step, Array<T> &out,
            ulong start, ulong end) override;

  // For this prox we cannot only override T call_single(T, step) const,
  // since we need the weights...
  void call_single(ulong i, const Array<T> &coeffs, T step,
                   Array<T> &out) const override;

  void call_single(ulong i, const Array<T> &coeffs, T step, Array<T> &out,
                   ulong n_times) const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void set_weights(SArrayTPtr weights) { this->weights = weights; }

 private:
  T call_single(T x, T step) const override;

  T call_single(T x, T step, ulong n_times) const override;

  T value_single(T x) const override;

  T call_single(T x, T step, T weight) const;

  T call_single(T x, T step, T weight, ulong n_times) const;

  T value_single(T x, T weight) const;
};

using ProxL1w = TProxL1w<double>;

using ProxL1wDouble = TProxL1w<double>;
using ProxL1wFloat = TProxL1w<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
