// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "tick/prox/prox_zero.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/solver/sdca.h"

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
  this->n_coeffs = model->get_n_coeffs();
  stored_variables_ready = false;
}

void SDCA::reset() {
  StoSolver::reset();
  set_starting_iterate();
}

void SDCA::solve() {
  if (!stored_variables_ready) {
    set_starting_iterate();
  }

  const SArrayULongPtr feature_index_map = model->get_sdca_index_map();
  const double scaled_l_l2sq = get_scaled_l_l2sq();

  const double _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);
  ulong start_t = t;

  for (t = start_t; t < start_t + epoch_size; ++t) {
    // Pick i uniformly at random
    ulong i = get_next_i();
    ulong feature_index = i;
    if (feature_index_map != nullptr) {
      feature_index = (*feature_index_map)[i];
    }

    // Maximize the dual coordinate i
    const double
      delta_dual_i =
      model->sdca_dual_min_i(feature_index, dual_vector[i], iterate, delta[i], scaled_l_l2sq);
    // Update the dual variable
    dual_vector[i] += delta_dual_i;

    // Keep the last ascent seen for warm-starting sdca_dual_min_i
    delta[i] = delta_dual_i;

    // Update the primal variable
    BaseArrayDouble features_i = model->get_features(feature_index);
    if (model->use_intercept()) {
      ArrayDouble primal_features = view(tmp_primal_vector, 0, features_i.size());
      primal_features.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
      tmp_primal_vector[model->get_n_features()] += delta_dual_i * _1_over_lbda_n;
    } else {
      tmp_primal_vector.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
    }
    // Call prox on the primal variable
    prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
  }
}

void SDCA::set_starting_iterate() {
  if (dual_vector.size() != rand_max)
    dual_vector = ArrayDouble(rand_max);

  dual_vector.init_to_zero();

  // If it is not ModelPoisReg, primal vector will be full of 0 as dual vector
  bool can_initialize_primal_to_zero = true;
  if (dynamic_cast<ModelPoisReg *>(model.get())) {
    std::shared_ptr<ModelPoisReg> casted_model = std::dynamic_pointer_cast<ModelPoisReg>(model);
    if (casted_model->get_link_type() == LinkType::identity) {
      can_initialize_primal_to_zero = false;
    }
  }

  if (can_initialize_primal_to_zero) {
    if (tmp_primal_vector.size() != n_coeffs)
      tmp_primal_vector = ArrayDouble(n_coeffs);

    if (iterate.size() != n_coeffs)
      iterate = ArrayDouble(n_coeffs);

    if (delta.size() != rand_max)
      delta = ArrayDouble(rand_max);

    iterate.init_to_zero();
    delta.init_to_zero();
    tmp_primal_vector.init_to_zero();
    stored_variables_ready = true;
  } else {
    set_starting_iterate(dual_vector);
  }
}

void SDCA::set_starting_iterate(ArrayDouble &dual_vector) {
  if (dual_vector.size() != rand_max) {
    TICK_ERROR("Starting iterate should be dual vector and have shape (" << rand_max << ", )");
  }

  if (!dynamic_cast<ProxZero *>(prox.get())) {
    TICK_ERROR("set_starting_iterate in SDCA might be call only if prox is ProxZero. Otherwise "
                 "we need to implement the Fenchel conjugate of the prox gradient");
  }

  if (iterate.size() != n_coeffs)
    iterate = ArrayDouble(n_coeffs);
  if (delta.size() != rand_max)
    delta = ArrayDouble(rand_max);

  this->dual_vector = dual_vector;
  model->sdca_primal_dual_relation(get_scaled_l_l2sq(), dual_vector, iterate);
  tmp_primal_vector = iterate;

  stored_variables_ready = true;
}
