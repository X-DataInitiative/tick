// License: BSD 3 clause

#include "tick/prox/prox_multi.h"

// ProxMulti can be instantiated with strength=0 only, since ProxMulti's strength is not used
ProxMulti::ProxMulti(std::vector<ProxPtr> proxs)
  : Prox(0, false), proxs(proxs) {
}

const std::string ProxMulti::get_class_name() const {
  return "ProxMulti";
}

double ProxMulti::value(const ArrayDouble &coeffs,
                        ulong start,
                        ulong end) {
  double val = 0;
  for (ProxPtr prox : proxs) {
    val += prox->value(coeffs);
  }
  return val;
}

void ProxMulti::call(const ArrayDouble &coeffs,
                     double step,
                     ArrayDouble &out,
                     ulong start,
                     ulong end) {
  // We need a copy
  ArrayDouble original_coeffs = coeffs;
  // Note for later: if all are ProxSeparable, we can avoid the copy...
  for (ProxPtr prox : proxs) {
    prox->call(original_coeffs, step, out);
    original_coeffs = out;
  }
}
