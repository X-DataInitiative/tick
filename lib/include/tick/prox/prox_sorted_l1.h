#ifndef LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_

// License: BSD 3 clause

#include "prox.h"

enum class WeightsType { bh = 0, oscar };

template <class T>
class TProxSortedL1 : public TProx<T> {
 protected:
  using TProx<T>::start;
  using TProx<T>::end;
  using TProx<T>::strength;

 protected:
  WeightsType weights_type;
  Array<T> weights;
  bool weights_ready;

  virtual void compute_weights(void);

  void prox_sorted_l1(const Array<T> &y, const Array<T> &strength,
                      Array<T> &x) const;

 public:
  TProxSortedL1(T strength, WeightsType weights_type, bool positive);

  TProxSortedL1(T strength, WeightsType weights_type, ulong start, ulong end,
                bool positive);

  std::string get_class_name() const override;

  T value(const Array<T> &coeffs, ulong start, ulong end) override;

  void call(const Array<T> &coeffs, T t, Array<T> &out, ulong start,
            ulong end) override;

  inline WeightsType get_weights_type() const { return weights_type; }

  inline void set_weights_type(WeightsType weights_type) {
    this->weights_type = weights_type;
    weights_ready = false;
  }

  inline T get_weight_i(ulong i) { return weights[i]; }

  void set_strength(T strength) override;

  void set_start_end(ulong start, ulong end) override;
};

using ProxSortedL1 = TProxSortedL1<double>;

using ProxSortedL1Double = TProxSortedL1<double>;
using ProxSortedL1Float = TProxSortedL1<float>;

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
