#ifndef LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
#define LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_

// License: BSD 3 clause

#include "prox_with_groups.h"

class ProxBinarsity : public TProxWithGroups<double, double> {
 protected:
  using TProxWithGroups<double, double>::is_synchronized;

 protected:
  std::unique_ptr<TProx<double, double> > build_prox(double strength, ulong start, ulong end, bool positive);

 public:
  ProxBinarsity(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  ProxBinarsity(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  std::string get_class_name() const;

  void call(const ArrayDouble &coeffs, double step, ArrayDouble &out,
            ulong start, ulong end);
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_BINARSITY_H_
