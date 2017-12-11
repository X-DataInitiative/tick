// License: BSD 3 clause

#include "tick/prox/prox_tv.h"
#include "tick/prox/prox_binarsity.h"

ProxBinarsity::ProxBinarsity(double strength,
                             SArrayULongPtr blocks_start,
                             SArrayULongPtr blocks_length,
                             bool positive)
    : ProxWithGroups(strength, blocks_start, blocks_length, positive) {}

ProxBinarsity::ProxBinarsity(double strength,
                             SArrayULongPtr blocks_start,
                             SArrayULongPtr blocks_length,
                             ulong start,
                             ulong end, bool positive)
    : ProxWithGroups(strength, blocks_start, blocks_length, start, end, positive) {}

std::unique_ptr<Prox> ProxBinarsity::build_prox(double strength, ulong start, ulong end, bool positive) {
  return std::unique_ptr<ProxTV>(new ProxTV(strength, start, end, positive));
}

const std::string ProxBinarsity::get_class_name() const {
  return "ProxBinarsity";
}

void ProxBinarsity::call(const ArrayDouble &coeffs,
                         double step,
                         ArrayDouble &out,
                         ulong start,
                         ulong end) {
  if (!is_synchronized) {
    synchronize_proxs();
  }
  for (auto &prox : proxs) {
    ulong start_k = prox->get_start();
    ulong end_k = prox->get_end();
    prox->call(coeffs, step, out, start_k, end_k);
    ArrayDouble out_block_k = view(out, start_k, end_k);
    double mean_k = out_block_k.sum() / (end_k - start_k);
    for (ulong j = 0; j < end_k - start_k; j++) {
      out_block_k[j] -= mean_k;
    }
  }
}
