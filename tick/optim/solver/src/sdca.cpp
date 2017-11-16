// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include "variants/hawkes_sdca_one_node.h"
#include "prox_l1.h"
#include "prox_zero.h"
#include "poisreg.h"
#include "sdca.h"

SDCA::SDCA(double l_l2sq,
           ulong epoch_size,
           double tol,
           RandType rand_type,
           BatchSize batch_size,
           ulong batch_number,
           int seed
) : StoSolver(epoch_size, tol, rand_type, seed),
    l_l2sq(l_l2sq), batch_size(batch_size), batch_number(batch_number) {
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

  const SArrayULongPtr feature_index_map_ptr = model->get_sdca_index_map();
  ArrayULong feature_index_map = feature_index_map_ptr == nullptr?
                                 ArrayULong(0): *feature_index_map_ptr;
  const double scaled_l_l2sq = get_scaled_l_l2sq();

  const double _1_over_lbda_n = 1 / (scaled_l_l2sq * rand_max);
  ulong start_t = t;

  for (t = start_t; t < start_t + epoch_size; ++t) {

    switch (batch_size) {
      case BatchSize::one :
        solve_batch_size_one(feature_index_map, scaled_l_l2sq, _1_over_lbda_n);
        break;
      case BatchSize::two :
        solve_batch_size_two(feature_index_map, scaled_l_l2sq, _1_over_lbda_n);
        break;
      case BatchSize::many :
          solve_batch_size_many(feature_index_map, scaled_l_l2sq, _1_over_lbda_n);
        break;
    }

    // Call prox on the primal variable
    prox->call(tmp_primal_vector, 1. / scaled_l_l2sq, iterate);
  }
}

void SDCA::solve_batch_size_one(ArrayULong &feature_index_map,
                                double scaled_l_l2sq, double _1_over_lbda_n) {
  // Pick i uniformly at random
  ulong i = get_next_i();
  ulong feature_index = i;
  if (feature_index_map.size() != 0) {
    feature_index = feature_index_map[i];
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
}

void SDCA::solve_batch_size_two(ArrayULong &feature_index_map,
                                double scaled_l_l2sq, double _1_over_lbda_n) {
  // Pick i, j uniformly at random
  ulong i = get_next_i();
  ulong j = get_next_i();
  while (j == i) j = get_next_i();

  ulong feature_index_i = i;
  ulong feature_index_j = j;
  if (feature_index_map.size() != 0) {
    feature_index_i = feature_index_map[i];
    feature_index_j = feature_index_map[j];
  }

  // Maximize the dual coordinates i and j
  double delta_dual_i, delta_dual_j;
  std::tie(delta_dual_i, delta_dual_j) =
    model->sdca_dual_min_ij(feature_index_i, feature_index_j,
                            dual_vector[i], dual_vector[j], iterate, scaled_l_l2sq);

  // Update the dual variable
  dual_vector[i] += delta_dual_i;
  dual_vector[j] += delta_dual_j;

  // Update the primal variable
  BaseArrayDouble features_i = model->get_features(feature_index_i);
  BaseArrayDouble features_j = model->get_features(feature_index_j);
  if (model->use_intercept()) {
    ArrayDouble primal_features = view(tmp_primal_vector, 0, features_i.size());
    primal_features.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
    primal_features.mult_incr(features_j, delta_dual_j * _1_over_lbda_n);
    tmp_primal_vector[model->get_n_features()] += (delta_dual_i + delta_dual_j) * _1_over_lbda_n;
  } else {
    tmp_primal_vector.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
    tmp_primal_vector.mult_incr(features_j, delta_dual_j * _1_over_lbda_n);
  }

  // add one step t
  t++;
}

void SDCA::solve_batch_size_many(ArrayULong &feature_index_map,
                                 double scaled_l_l2sq, double _1_over_lbda_n) {
  // Pick indices uniformly at random
  ArrayULong indices(batch_number);
  indices.fill(rand_max + 1);

  ArrayDouble duals(batch_number);
  for (ulong i = 0; i < batch_number; ++i) {
    ulong try_i = get_next_i();
    while (indices.contains(try_i)) try_i = get_next_i();
    indices[i] = try_i;
    duals[i] = dual_vector[indices[i]];
  }

  ArrayULong feature_indices(batch_number);
  for (ulong i = 0; i < batch_number; ++i)
     feature_indices[i] = feature_index_map.size() != 0? feature_index_map[indices[i]]: indices[i];

  // Maximize the dual coordinates
// // Uncomment to compare solve_batch_size_many and solve_batch_size_two
//  ArrayDouble delta_duals(2);
//  if (batch_number == 2) {
//    double delta_dual_i, delta_dual_j;
//    std::tie(delta_dual_i, delta_dual_j) =
//      model->sdca_dual_min_ij(feature_indices[0], feature_indices[1],
//                              duals[0], duals[1], iterate, scaled_l_l2sq);
//    delta_duals[0] = delta_dual_i;
//    delta_duals[1] = delta_dual_j;
//  }
//  else {
//    delta_duals =
//      model->sdca_dual_min_many(feature_indices, duals, iterate, scaled_l_l2sq);
//  }

  ArrayDouble delta_duals =
    model->sdca_dual_min_many(feature_indices, duals, iterate, scaled_l_l2sq);

  // Update the dual variable
  for (ulong i = 0; i < batch_number; ++i) {
    dual_vector[indices[i]] += delta_duals[i];

    // Update the primal variable
    BaseArrayDouble features_i = model->get_features(feature_indices[i]);
    if (model->use_intercept()) {
      ArrayDouble primal_features = view(tmp_primal_vector, 0, features_i.size());
      primal_features.mult_incr(features_i, delta_duals[i] * _1_over_lbda_n);
      tmp_primal_vector[model->get_n_features()] += delta_duals[i] * _1_over_lbda_n;
    } else {
      tmp_primal_vector.mult_incr(features_i, delta_duals[i] * _1_over_lbda_n);
    }
  }

  // add few steps to t
  t += batch_number - 1;
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
  if (dynamic_cast<ModelHawkesSDCAOneNode *>(model.get())) {
    can_initialize_primal_to_zero = false;
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

  if (!(dynamic_cast<ProxZero *>(prox.get()) || dynamic_cast<ProxL1 *>(prox.get()))) {
    TICK_ERROR("set_starting_iterate in SDCA might be call only if prox is ProxZero. Otherwise "
                 "we need to implement the Fenchel conjugate of the prox gradient");
  }

  if (iterate.size() != n_coeffs)
    iterate = ArrayDouble(n_coeffs);
  if (delta.size() != rand_max)
    delta = ArrayDouble(rand_max);

  this->dual_vector = dual_vector;
  model->sdca_primal_dual_relation(get_scaled_l_l2sq(), dual_vector, iterate);
  prox->call(iterate, 1. / get_scaled_l_l2sq(), iterate);
  tmp_primal_vector = iterate;

  stored_variables_ready = true;
}
