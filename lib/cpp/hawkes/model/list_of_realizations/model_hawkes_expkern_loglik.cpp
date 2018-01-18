// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_loglik.h"

ModelHawkesExpKernLogLik::ModelHawkesExpKernLogLik(
  const double decay, const int max_n_threads) :
  ModelHawkesLogLik(max_n_threads), decay(decay) {}

ulong ModelHawkesExpKernLogLik::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
