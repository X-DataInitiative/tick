// License: BSD 3 clause


#include "tick/hawkes/model/variants/hawkes_fixed_sumexpkern_loglik_list.h"

ModelHawkesFixedSumExpKernLogLikList::ModelHawkesFixedSumExpKernLogLikList(
  const ArrayDouble &decays, const int max_n_threads) :
  ModelHawkesFixedKernLogLikList(max_n_threads), decays(decays) {}

ulong ModelHawkesFixedSumExpKernLogLikList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays();
}
