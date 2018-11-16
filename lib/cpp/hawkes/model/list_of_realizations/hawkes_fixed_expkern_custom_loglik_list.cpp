// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/hawkes_fixed_expkern_custom_loglik_list.h"

ModelHawkesFixedExpKernCustomLogLikList::ModelHawkesFixedExpKernCustomLogLikList(
  const double &_decay, const ulong _MaxN_of_f, const int max_n_threads) :
        ModelHawkesCustomLogLikList(max_n_threads), MaxN_of_f(_MaxN_of_f), decay(_decay) {}

ulong ModelHawkesFixedExpKernCustomLogLikList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes + n_nodes * (MaxN_of_f);
}
