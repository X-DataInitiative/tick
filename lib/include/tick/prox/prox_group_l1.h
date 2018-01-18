#ifndef LIB_INCLUDE_TICK_PROX_PROX_GROUP_L1_H_
#define LIB_INCLUDE_TICK_PROX_PROX_GROUP_L1_H_

// License: BSD 3 clause

#include "prox_with_groups.h"

class ProxGroupL1 : public ProxWithGroups {
 protected:
  std::unique_ptr<Prox> build_prox(double strength, ulong start, ulong end, bool positive) final;

 public:
  ProxGroupL1(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                bool positive);

  ProxGroupL1(double strength, SArrayULongPtr blocks_start, SArrayULongPtr blocks_length,
                ulong start, ulong end, bool positive);

  const std::string get_class_name() const final;
};

#endif  // LIB_INCLUDE_TICK_PROX_PROX_GROUP_L1_H_
