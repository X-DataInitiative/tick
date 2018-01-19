// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_loglik.h"

ModelHawkesSumExpKernLogLik::ModelHawkesSumExpKernLogLik(
  const ArrayDouble &decays, const int max_n_threads) :
  ModelHawkesLogLik(max_n_threads), decays(decays) {}

ulong ModelHawkesSumExpKernLogLik::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays();
}
