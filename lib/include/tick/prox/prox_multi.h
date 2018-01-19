
#ifndef LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
#define LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_

// License: BSD 3 clause

#include "prox.h"

// TODO: this requires some work. ProxMulti should have the standard
// TODO: prox API, with a set_strength, and things like that

class ProxMulti : public Prox {
 protected:
  std::vector<ProxPtr> proxs;

 public:
  explicit ProxMulti(std::vector<ProxPtr> proxs);

  const std::string get_class_name() const override;

  double value(const ArrayDouble &coeffs, ulong start, ulong end) override;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out, ulong start,
            ulong end) override;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_MULTI_H_
