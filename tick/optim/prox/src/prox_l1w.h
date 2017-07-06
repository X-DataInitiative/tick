#ifndef TICK_OPTIM_PROX_SRC_PROX_L1W_H_
#define TICK_OPTIM_PROX_SRC_PROX_L1W_H_

// License: BSD 3 clause

#include "base.h"
#include "prox_separable.h"

class ProxL1w : public ProxSeparable {
 protected:
  // Weights for L1 penalization
  SArrayDoublePtr weights;

 public:
  ProxL1w(double strength, SArrayDoublePtr weights, bool positive);

  ProxL1w(double strength, SArrayDoublePtr weights, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  // For this prox we cannot only override double call_single(double, step) const,
  // since we need the weights...
  void call_single(ulong i, const ArrayDouble &coeffs, double step,
                   ArrayDouble &out) const override;

  void call_single(ulong i, const ArrayDouble &coeffs, double step,
                   ArrayDouble &out, ulong n_times) const override;

  double value_single(ulong i,
                      const ArrayDouble &coeffs) const override;

  void set_weights(SArrayDoublePtr weights) {
    this->weights = weights;
  }
};

#endif  // TICK_OPTIM_PROX_SRC_PROX_L1W_H_
