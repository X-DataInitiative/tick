//
// Created by Martin Bompaire on 22/10/15.
//

#include <prox_l2sq.h>
#include "sdca.h"

SDCA::SDCA(double l_l2sq,
           ulong epoch_size,
           double tol,
           RandType rand_type,
           int seed
) : StoSolver(epoch_size, tol, rand_type, seed), l_l2sq(l_l2sq) {
  stored_variables_ready = false;
}

void SDCA::set_model(ModelPtr model) {
  StoSolver::set_model(model);
  this->model = model;
  stored_variables_ready = false;
}

void SDCA::reset() {
  StoSolver::reset();
  init_stored_variables();
}

void SDCA::init_stored_variables() {
  n_samples = model->get_n_samples();
  n_coeffs = model->get_n_coeffs();

  if (dual_vector.size() != n_samples)
    dual_vector = ArrayDouble(n_samples);

  if (delta.size() != n_samples)
    delta = ArrayDouble(n_samples);

  if (tmp_primal_vector.size() != n_coeffs)
    tmp_primal_vector = ArrayDouble(n_coeffs);

  dual_vector.init_to_zero();
  delta.init_to_zero();
  tmp_primal_vector.init_to_zero();
  stored_variables_ready = true;
}

void SDCA::solve() {
  if (!stored_variables_ready) {
    init_stored_variables();
  }

  ulong i;
  double delta_i;
  double _1_over_lbda_n = 1 / (l_l2sq * n_samples);
  ulong start_t = t;

  for (t = start_t; t < start_t + epoch_size; ++t) {
    // Pick i uniformly at random
    i = get_next_i();

    // Maximize the dual coordinate i
    delta_i = model->sdca_dual_min_i(i, dual_vector, iterate, delta, l_l2sq);

    // Update the dual variable
    dual_vector[i] += delta_i;

    // Keep the last ascent seen for warm-starting sdca_dual_min_i
    delta[i] = delta_i;

    // Update the primal variable
    BaseArrayDouble features_i = model->get_features(i);

    if (model->use_intercept()) {
      ArrayDouble primal_features = view(tmp_primal_vector, 0, features_i.size());
      primal_features.mult_incr(features_i, delta_i * _1_over_lbda_n);
      tmp_primal_vector[model->get_n_features()] += delta_i * _1_over_lbda_n;
    } else {
      tmp_primal_vector.mult_incr(features_i, delta_i * _1_over_lbda_n);
    }

    // Call prox on the primal variable
    prox->call(tmp_primal_vector, 1. / l_l2sq, iterate);
  }
}
