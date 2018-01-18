#ifndef LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
#define LIB_INCLUDE_TICK_PROX_PROX_L1W_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "prox_separable.h"

class ProxL1w : public ProxSeparable {
 protected:
  // Weights for L1 penalization
  SArrayDoublePtr weights;

 public:
  ProxL1w(double strength, SArrayDoublePtr weights, bool positive);

  ProxL1w(double strength, SArrayDoublePtr weights, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

  void call(const ArrayDouble &coeffs, const double step, ArrayDouble &out,
            ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, const ArrayDouble &step, ArrayDouble &out,
            ulong start, ulong end) override;

  // For this prox we cannot only override double call_single(double, step) const,
  // since we need the weights...
  void call_single(ulong i, const ArrayDouble &coeffs, double step,
                   ArrayDouble &out) const override;

  void call_single(ulong i, const ArrayDouble &coeffs, double step,
                   ArrayDouble &out, ulong n_times) const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void set_weights(SArrayDoublePtr weights) {
    this->weights = weights;
  }

 private:
  double call_single(double x, double step) const override;

  double call_single(double x, double step, ulong n_times) const override;

  double value_single(double x) const override;

  double call_single(double x, double step, double weight) const;

  double call_single(double x, double step, double weight, ulong n_times) const;

  double value_single(double x, double weight) const;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_L1W_H_
