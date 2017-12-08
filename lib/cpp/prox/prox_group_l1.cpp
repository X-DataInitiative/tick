// License: BSD 3 clause

#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_group_l1.h"

ProxGroupL1::ProxGroupL1(double strength,
                         SArrayULongPtr blocks_start,
                         SArrayULongPtr blocks_length,
                         bool positive)
    : ProxWithGroups(strength, blocks_start, blocks_length, positive) {}

ProxGroupL1::ProxGroupL1(double strength,
                         SArrayULongPtr blocks_start,
                         SArrayULongPtr blocks_length,
                         ulong start,
                         ulong end, bool positive)
    : ProxWithGroups(strength, blocks_start, blocks_length, start, end, positive) {}

const std::string ProxGroupL1::get_class_name() const {
  return "ProxGroupL1";
}

std::unique_ptr<Prox> ProxGroupL1::build_prox(double strength, ulong start, ulong end, bool positive) {
  return std::unique_ptr<ProxL2>(new ProxL2(strength, start, end, positive));
}
