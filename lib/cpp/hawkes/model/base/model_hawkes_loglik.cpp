// License: BSD 3 clause


#include "tick/hawkes/model/base/model_hawkes_loglik.h"

ModelHawkesLogLik::ModelHawkesLogLik(const int max_n_threads) :
  ModelHawkesList(max_n_threads, 0) {}

void ModelHawkesLogLik::incremental_set_data(
  const SArrayDoublePtrList1D &timestamps, double end_time) {
  weights_computed = false;
  if (model_list.empty()) {
    set_n_nodes(timestamps.size());

    n_realizations = 0;
    end_times = VArrayDouble::new_ptr(0);
    n_jumps_per_realization = VArrayULong::new_ptr(0);
    n_jumps_per_node = SArrayULong::new_ptr(n_nodes);
    n_jumps_per_node->init_to_zero();
  } else {
    if (n_nodes != timestamps.size()) {
      TICK_ERROR("Your realization should have " << n_nodes << " nodes but has "
                                                 << timestamps.size() << ".");
    }
  }

  n_realizations += 1;
  end_times->append1(end_time);

  ulong n_total_jumps = 0;
  for (ulong i = 0; i < n_nodes; ++i) {
    n_total_jumps += timestamps[i]->size();
    (*n_jumps_per_node)[i] += timestamps[i]->size();
  }
  n_jumps_per_realization->append1(n_total_jumps);

  auto model = build_model(get_n_threads());
  model->set_data(timestamps, end_time);
  model->compute_weights();
  model_list.push_back(std::move(model));

  weights_computed = true;
}

void ModelHawkesLogLik::compute_weights() {
  if (!model_list.empty() && timestamps_list.size() != model_list.size()) {
    TICK_ERROR("Cannot compute weights as timestamps have not been stored. "
                 "Did you use incremental_fit?");
  }
  model_list = std::vector<std::unique_ptr<ModelHawkesLogLikSingle> >(n_realizations);

  for (ulong r = 0; r < n_realizations; ++r) {
    model_list[r] = build_model(1);
    model_list[r]->set_data(timestamps_list[r], (*end_times)[r]);
    model_list[r]->allocate_weights();
  }

  parallel_run(get_n_threads(), n_realizations * n_nodes,
               &ModelHawkesLogLik::compute_weights_i_r, this);

  for (auto &model : model_list) {
    model->weights_computed = true;
  }
  weights_computed = true;
}

std::tuple<ulong, ulong> ModelHawkesLogLik::get_realization_node(ulong i_r) {
  const ulong r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;
  return std::make_tuple(r, i);
}

void ModelHawkesLogLik::compute_weights_i_r(const ulong i_r) {
  ulong r, i;
  std::tie(r, i) = get_realization_node(i_r);
  model_list[r]->compute_weights_dim_i(i);
}

double ModelHawkesLogLik::loss_i_r(const ulong i_r, const ArrayDouble &coeffs) {
  ulong r, i;
  std::tie(r, i) = get_realization_node(i_r);

  return model_list[r]->loss_dim_i(i, coeffs);
}

double ModelHawkesLogLik::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  return parallel_map_additive_reduce(
    get_n_threads(), n_realizations * n_nodes,
    &ModelHawkesLogLik::loss_i_r, this, coeffs) / get_n_total_jumps();
}

double ModelHawkesLogLik::loss_i(const ulong i, const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  const auto r_i = sampled_i_to_realization(i);
  return model_list[r_i.first]->loss_i(r_i.second, coeffs);
}

void ModelHawkesLogLik::grad_i_r(const ulong i_r,
                                 ArrayDouble &out,
                                 const ArrayDouble &coeffs) {
  ulong r, i;
  std::tie(r, i) = get_realization_node(i_r);

  ArrayDouble tmp_grad_i(get_n_coeffs());
  tmp_grad_i.init_to_zero();
  model_list[r]->grad_dim_i(i, coeffs, tmp_grad_i);
  out.mult_incr(tmp_grad_i, 1.);
}

void ModelHawkesLogLik::grad(const ArrayDouble &coeffs, ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  out.init_to_zero();
  parallel_map_array<ArrayDouble>(
    get_n_threads(), n_realizations * n_nodes,
    [](ArrayDouble &r, const ArrayDouble &s) { r.mult_incr(s, 1.0); },
    &ModelHawkesLogLik::grad_i_r,
    this, out, coeffs);
  out /= get_n_total_jumps();
}

void ModelHawkesLogLik::grad_i(const ulong i, const ArrayDouble &coeffs,
                               ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  const auto r_i = sampled_i_to_realization(i);
  model_list[r_i.first]->grad_i(r_i.second, coeffs, out);
}

double ModelHawkesLogLik::loss_and_grad(const ArrayDouble &coeffs,
                                        ArrayDouble &out) {
  // TODO(svp) create parallel_map_array_reduce_result
  // In order to sum output (losses) and keep reducing gradients
  // This could allow us to use loss_and_grad_dim_i
  grad(coeffs, out);
  return loss(coeffs);
}

double ModelHawkesLogLik::hessian_norm_i_r(const ulong i_r,
                                           const ArrayDouble &coeffs,
                                           const ArrayDouble &vector) {
  ulong r, i;
  std::tie(r, i) = get_realization_node(i_r);

  return model_list[r]->hessian_norm_dim_i(i, coeffs, vector);
}

double ModelHawkesLogLik::hessian_norm(const ArrayDouble &coeffs,
                                       const ArrayDouble &vector) {
  if (!weights_computed) compute_weights();
  return parallel_map_additive_reduce(
    get_n_threads(), n_realizations * n_nodes,
    &ModelHawkesLogLik::hessian_norm_i_r, this, coeffs, vector)
    / get_n_total_jumps();
}

void ModelHawkesLogLik::hessian_i_r(const ulong i_r,
                                    const ArrayDouble &coeffs,
                                    ArrayDouble &out) {
  ulong r, i;
  std::tie(r, i) = get_realization_node(i_r);

  model_list[r]->hessian_i(i, coeffs, out);
}

void ModelHawkesLogLik::hessian(const ArrayDouble &coeffs,
                                ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  parallel_run(
    get_n_threads(), n_realizations * n_nodes,
    &ModelHawkesLogLik::hessian_i_r, this, coeffs, out);
  out /= get_n_total_jumps();
}

std::pair<ulong, ulong> ModelHawkesLogLik::sampled_i_to_realization(
  const ulong sampled_i) {
  ulong cum_n_jumps = 0;
  for (ulong r = 0; r < n_realizations; r++) {
    cum_n_jumps += (*n_jumps_per_realization)[r];
    if (sampled_i < cum_n_jumps) {
      const ulong i_in_realization_r = sampled_i - cum_n_jumps + (*n_jumps_per_realization)[r];
      return std::pair<ulong, ulong>(r, i_in_realization_r);
    }
  }
  TICK_ERROR("sampled_i out of range");
}

ulong ModelHawkesLogLik::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
