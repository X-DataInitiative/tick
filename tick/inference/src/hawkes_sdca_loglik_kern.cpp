// License: BSD 3 clause

#include "variants/hawkes_sdca_one_node.h"
#include "hawkes_fixed_expkern_loglik.h"
#include "hawkes_sdca_loglik_kern.h"

HawkesSDCALoglikKern::HawkesSDCALoglikKern(const ArrayDouble &decays, double l_l2sq,
                                           int max_n_threads, double tol,
                                           RandType rand_type, int seed)
  : ModelHawkesList(max_n_threads, optimization_level),
    weights_allocated(false), l_l2sq(l_l2sq), tol(tol), rand_type(rand_type), seed(seed) {
  set_decays(decays);
}

HawkesSDCALoglikKern::HawkesSDCALoglikKern(const double decay, double l_l2sq,
                                           int max_n_threads, double tol,
                                           RandType rand_type, int seed)
  : ModelHawkesList(max_n_threads, optimization_level),
    weights_allocated(false), l_l2sq(l_l2sq), tol(tol), rand_type(rand_type), seed(seed) {
  ArrayDouble decays {decay};
  set_decays(decays);
}

void HawkesSDCALoglikKern::allocate_weights() {
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    ulong n_jumps_node_i = (*n_jumps_per_node)[i];
    g[i] = ArrayDouble2d(n_jumps_node_i, get_n_coeffs_per_node(), nullptr);
    G[i] = ArrayDouble(get_n_coeffs_per_node());
  }

  weights_allocated = true;
}

void HawkesSDCALoglikKern::compute_weights() {
  if (!weights_allocated) allocate_weights();

  // We will need each thread to store its value of G in buffers that we will merge afterwards
  auto G_buffer = std::make_shared<ArrayDouble2dList1D>(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      (*G_buffer)[i] = ArrayDouble2d(n_realizations, get_n_coeffs_per_node(), nullptr);
      (*G_buffer)[i].init_to_zero();
    }
  }

  for (ulong i = 0; i < n_nodes; i++) {
    g[i].init_to_zero();
    G[i].init_to_zero();
  }

  parallel_run(get_n_threads(), n_nodes * n_realizations,
               &HawkesSDCALoglikKern::compute_weights_dim_i, this, G_buffer);

  for (ulong i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      G[i].mult_incr(view_row((*G_buffer)[i], r), 1);
    }
  }

  weights_computed = true;

  synchronize_sdca();
}

void HawkesSDCALoglikKern::synchronize_sdca() {
  if (!weights_computed) compute_weights();
  sdca_list = std::vector<SDCA>();
  for (ulong i = 0; i < n_nodes; ++i) {
    auto model = std::make_shared<ModelHawkesSDCAOneNode>(g[i], G[i], get_n_samples());

    const ulong epoch_size = (*n_jumps_per_node)[i];
    SDCA sdca(l_l2sq, epoch_size, tol, rand_type, seed);

    sdca.set_model(model);
    sdca.set_rand_max(epoch_size);

    sdca_list.push_back(sdca);
  }
}

void HawkesSDCALoglikKern::compute_weights_dim_i(const ulong i_r,
                                                 std::shared_ptr<ArrayDouble2dList1D> G_buffer) {
  const auto r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;

  const ArrayDouble t_i = view(*timestamps_list[r][i]);
  ArrayDouble2d g_i = view(g[i]);
  ArrayDouble G_i_r = view_row((*G_buffer)[i], r);

  const double end_time = (*end_times)[r];
  const ulong n_jumps_i = t_i.size();
  ulong start_row = 0;
  for (ulong smaller_r = 0; smaller_r < r; ++smaller_r) {
    start_row += timestamps_list[smaller_r][i]->size();
  }

  for (ulong j = 0; j < n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps_list[r][j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const ulong row_k = start_row + k;
      const double t_i_k = k < n_jumps_i ? t_i[k] : end_time;

      for (ulong u = 0; u < get_n_decays(); ++u) {
        const ulong col_ju = 1 + j * get_n_decays() + u;
        const double decay_u = decays[u];
        if (k > 0) {
          const double ebt = std::exp(-decay_u * (t_i_k - t_i[k - 1]));
          if (k < n_jumps_i) {
            g_i(row_k, col_ju) = g_i(row_k - 1, col_ju) * ebt;
          }
          G_i_r[col_ju] += g_i(row_k - 1, col_ju) * (1 - ebt) / decay_u;
        } else {
          g_i(row_k, col_ju) = 0;
          G_i_r[col_ju] = 0;
        }
      }

      while ((ij < t_j.size()) && (t_j[ij] < t_i_k)) {
        for (ulong u = 0; u < get_n_decays(); ++u) {
          const ulong col_ju = 1 + j * get_n_decays() + u;
          const double decay_u = decays[u];
          const double ebt = std::exp(-decay_u * (t_i_k - t_j[ij]));
          if (k < n_jumps_i) g_i(row_k, col_ju) += decay_u * ebt;
          G_i_r[col_ju] += 1 - ebt;
        }
        ij++;
      }
      // fill mu part
      if (k < n_jumps_i) g_i(row_k, 0) = 1.;
    }
  }
  G_i_r[0] = end_time;
}

// The main method for performing one iteration
void HawkesSDCALoglikKern::solve() {
  if (!weights_computed) compute_weights();
  return parallel_run(get_n_threads(), n_nodes,
                      &HawkesSDCALoglikKern::solve_dim_i, this);
}

void HawkesSDCALoglikKern::solve_dim_i(const ulong i) {
  sdca_list[i].solve();
}

double HawkesSDCALoglikKern::current_dual_objective() {
  if (!weights_computed) compute_weights();
  return parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                      &HawkesSDCALoglikKern::current_dual_objective_dim_i, this);
}
double HawkesSDCALoglikKern::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  return parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                      &HawkesSDCALoglikKern::loss_dim_i, this,
                                      coeffs);
}

double HawkesSDCALoglikKern::current_dual_objective_dim_i(const ulong i) const {
  ArrayDouble dual_vector = sdca_list[i].get_dual_vector_view();
  auto model_ptr = sdca_list[i].get_model();
  return model_ptr->dual_objective(l_l2sq, dual_vector);
}

double HawkesSDCALoglikKern::loss_dim_i(const ulong i, const ArrayDouble &coeffs) const {
  auto model_ptr = sdca_list[i].get_model();

  ArrayDouble local_coeffs(get_n_coeffs_per_node());
  local_coeffs[0] = coeffs[i];

  const ulong n_coeffs_alpha = get_n_coeffs_per_node() - 1;
  ArrayDouble coeffs_alpha = view(coeffs, n_nodes + i * n_coeffs_alpha,
                                  n_nodes + (i + 1) * n_coeffs_alpha);
  view(local_coeffs, 1).mult_fill(coeffs_alpha, 1);
  return model_ptr->loss(local_coeffs);
}

SArrayDoublePtr HawkesSDCALoglikKern::get_decays() const {
  ArrayDouble copy = decays;
  return copy.as_sarray_ptr();
}

void HawkesSDCALoglikKern::set_decays(const ArrayDouble &decays) {
  if (decays.min() <= 0) {
    TICK_ERROR("all decays must be positive, received " << decays);
  }
  this->decays = decays;
  weights_computed = false;
}

SArrayDoublePtr HawkesSDCALoglikKern::get_iterate()  {

  if (sdca_list.size() != n_nodes){
    SArrayDoublePtr zero_iterate = SArrayDouble::new_ptr(get_n_coeffs());
    zero_iterate->init_to_zero();
    return zero_iterate;
  }

  ArrayDouble iterate(get_n_coeffs());
  for (ulong i = 0; i < n_nodes; ++i) {
    ArrayDouble sdca_iterate(get_n_coeffs_per_node());
    sdca_list[i].get_iterate(sdca_iterate);
    iterate[i] = sdca_iterate[0];

    const ulong alpha_start = n_nodes + (get_n_coeffs_per_node() - 1) * i;
    for (ulong j = 0; j < get_n_coeffs_per_node() - 1; ++j) {
      iterate[alpha_start + j] = sdca_iterate[1 + j];
    }
  }
  return iterate.as_sarray_ptr();
}

SArrayDoublePtr HawkesSDCALoglikKern::get_dual_iterate() {
  if (sdca_list.size() != n_nodes){
    SArrayDoublePtr zero_iterate = SArrayDouble::new_ptr(get_n_total_jumps());
    zero_iterate->init_to_zero();
    return zero_iterate;
  }

  ArrayDouble dual_iterate(get_n_total_jumps());
  ulong position = 0;
  for (ulong i = 0; i < n_nodes; ++i) {
    ulong n_jumps_node_i = (*get_n_jumps_per_node())[i];
    ArrayDouble sdca_dual_iterate = sdca_list[i].get_dual_vector_view();
    ArrayDouble dual_iterate_view = view(dual_iterate, position, position + n_jumps_node_i);
    dual_iterate_view.mult_fill(sdca_dual_iterate, 1);
    position += n_jumps_node_i;
  }
  return dual_iterate.as_sarray_ptr();
}