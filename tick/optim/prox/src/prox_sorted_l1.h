//
// Created by Stéphane GAIFFAS on 30/12/2015.
//

#ifndef TICK_OPTIM_PROX_SRC_PROX_SORTED_L1_H_
#define TICK_OPTIM_PROX_SRC_PROX_SORTED_L1_H_

#include "prox.h"

enum class WeightsType {
  bh = 0,
  oscar
};

class ProxSortedL1 : public Prox {
 protected:
  bool positive;
  WeightsType weights_type;
  ArrayDouble weights;

  // Are the weights ready ?
  bool weights_ready;

  virtual void compute_weights(void);

  void prox_sorted_l1(ArrayDouble &y, ArrayDouble &strength, ArrayDouble &x) const;

 public:
  ProxSortedL1(double strength, WeightsType weights_type, bool positive);

  ProxSortedL1(double strength, WeightsType weights_type,
               ulong start, ulong end, bool positive);

  const std::string get_class_name() const;

  double _value(ArrayDouble &coeffs,
                ulong start,
                ulong end);

  void _call(ArrayDouble &coeffs,
             double t,
             ArrayDouble &out,
             ulong start,
             ulong end);

  inline WeightsType get_weights_type() const {
      return weights_type;
  }

  inline void set_weights_type(WeightsType weights_type) {
      this->weights_type = weights_type;
      weights_ready = false;
  }

  inline bool get_positive() const {
      return positive;
  }

  inline void set_positive(bool positive) {
      this->positive = positive;
  }

  inline double get_weight_i(ulong i) {
      return weights[i];
  }

  virtual void set_strength(double strength) {
    if (strength != this->strength) {
      weights_ready = false;
    }
    this->strength = strength;
  }

  // We overload set_start_end here, since we'd need to update weights when they're changed
  inline void set_start_end(ulong start, ulong end) {
      if ((start != this->start) || (end != this->end)) {
          // If we change the range, we need to compute again the weights
          weights_ready = false;
      }
      this->has_range = true;
      this->start = start;
      this->end = end;
  }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_SORTED_L1_H_
