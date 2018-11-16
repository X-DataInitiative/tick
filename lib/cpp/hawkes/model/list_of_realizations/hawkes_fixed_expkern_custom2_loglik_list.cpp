// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/hawkes_fixed_expkern_custom2_loglik_list.h"

ModelHawkesFixedExpKernCustomType2LogLikList::ModelHawkesFixedExpKernCustomType2LogLikList(
  const double &_decay, const ulong _MaxN, const int max_n_threads) :
        ModelHawkesCustomLogLikList(max_n_threads), MaxN(_MaxN), decay(_decay) {}

ulong ModelHawkesFixedExpKernCustomType2LogLikList::get_n_coeffs() const {
  return n_nodes * MaxN + n_nodes * n_nodes;
}
