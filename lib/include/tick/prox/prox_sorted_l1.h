#ifndef LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_

// License: BSD 3 clause

#include "prox.h"

enum class WeightsType {
  bh = 0,
  oscar
};

class ProxSortedL1 : public Prox {
 protected:
  WeightsType weights_type;
  ArrayDouble weights;
  bool weights_ready;

  virtual void compute_weights(void);

  void prox_sorted_l1(const ArrayDouble &y, const ArrayDouble &strength,
                      ArrayDouble &x) const;

 public:
  ProxSortedL1(double strength, WeightsType weights_type,
               bool positive);

  ProxSortedL1(double strength, WeightsType weights_type, ulong start,
               ulong end, bool positive);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double t, ArrayDouble &out, ulong start,
            ulong end) override;

  inline WeightsType get_weights_type() const {
    return weights_type;
  }

  inline void set_weights_type(WeightsType weights_type) {
    this->weights_type = weights_type;
    weights_ready = false;
  }

  inline double get_weight_i(ulong i) {
    return weights[i];
  }

  void set_strength(double strength) override;

  void set_start_end(ulong start, ulong end) override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_SORTED_L1_H_
