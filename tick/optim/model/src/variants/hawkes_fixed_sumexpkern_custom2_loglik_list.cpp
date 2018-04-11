// License: BSD 3 clause


#include "hawkes_fixed_sumexpkern_custom2_loglik_list.h"

ModelHawkesFixedSumExpKernCustomType2LogLikList::ModelHawkesFixedSumExpKernCustomType2LogLikList(
  const ArrayDouble &decays, const ulong _MaxN, const int max_n_threads) :
        ModelHawkesCustomLogLikList(max_n_threads), MaxN(_MaxN), decays(decays) {}

ulong ModelHawkesFixedSumExpKernCustomType2LogLikList::get_n_coeffs() const {
  return n_nodes * MaxN + n_nodes * n_nodes * get_n_decays();
}
