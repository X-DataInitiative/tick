#ifndef TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
#define TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_

// License: BSD 3 clause

#include "prox_separable.h"

class ProxL2Sq : public ProxSeparable {
 public:
  ProxL2Sq(double strength = 0, bool positive = false);

  ProxL2Sq(double strength, ulong start, ulong end, bool positive);

  const std::string get_class_name() const override;

 private:
  double value_single(double x) const override;

  double call_single(double x, double step) const override;

  // Repeat n_times the prox on coordinate i
  double call_single(double x, double step, ulong n_times) const override;

 public:
  template<class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("ProxSeparable", cereal::base_class<ProxSeparable>(this)));
  }
};

CEREAL_REGISTER_TYPE(ProxL2Sq)

#endif  // TICK_OPTIM_PROX_SRC_PROX_L2SQ_H_
