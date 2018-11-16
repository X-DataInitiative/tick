// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/custom_basic_loglik_list.h"

ModelCustomBasicLogLikList::ModelCustomBasicLogLikList(
  const double &_decay, const ulong _MaxN_of_f, const int max_n_threads) :
        ModelHawkesCustomLogLikList(max_n_threads), MaxN_of_f(_MaxN_of_f), decay(_decay) {}

ulong ModelCustomBasicLogLikList::get_n_coeffs() const {
  return n_nodes * MaxN_of_f;
}
