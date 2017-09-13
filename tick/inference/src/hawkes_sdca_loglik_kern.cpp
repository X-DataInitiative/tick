// License: BSD 3 clause

#include "variants/hawkes_sdca_one_node.h"
#include "hawkes_fixed_expkern_loglik.h"
#include "hawkes_sdca_loglik_kern.h"

HawkesSDCALoglikKern::HawkesSDCALoglikKern(double decay, double l_l2sq,
                                           int max_n_threads, double tol,
                                           RandType rand_type, int seed)
  : ModelHawkesList(max_n_threads, optimization_level),
    weights_allocated(false), l_l2sq(l_l2sq), tol(tol), rand_type(rand_type), seed(seed) {
  set_decay(decay);
}

void HawkesSDCALoglikKern::allocate_weights() {
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    ulong n_jumps_node_i = (*n_jumps_per_node)[i];
    g[i] = ArrayDouble2d(n_jumps_node_i, 1 + n_nodes, nullptr);
    G[i] = ArrayDouble(1 + n_nodes);
  }

  weights_allocated = true;
}

void HawkesSDCALoglikKern::compute_weights() {
  if (!weights_allocated) allocate_weights();

  // We will need each thread to store its value of G in buffers that we will merge afterwards
  auto G_buffer = std::make_shared<ArrayDouble2dList1D>(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      (*G_buffer)[i] = ArrayDouble2d(n_realizations, 1ul + n_nodes, nullptr);
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
    const ulong col_j = 1 + j;
    const ArrayDouble t_j = view(*timestamps_list[r][j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const ulong row_k = start_row + k;

      const double t_i_k = k < n_jumps_i ? t_i[k] : end_time;
      if (k > 0) {
        const double ebt = std::exp(-decay * (t_i_k - t_i[k - 1]));
        if (k < n_jumps_i) {
          g_i(row_k, col_j) = g_i(row_k - 1, col_j) * ebt;
        }
        G_i_r[col_j] += g_i(row_k - 1, col_j) * (1 - ebt) / decay;
      } else {
        g_i(row_k, col_j) = 0;
        G_i_r[col_j] = 0;
      }

      while ((ij < t_j.size()) && (t_j[ij] < t_i_k)) {
        const double ebt = std::exp(-decay * (t_i_k - t_j[ij]));
        if (k < n_jumps_i) g_i(row_k, col_j) += decay * ebt;
        G_i_r[col_j] += 1 - ebt;
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
  return parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                      &HawkesSDCALoglikKern::current_dual_objective_dim_i, this);
}
double HawkesSDCALoglikKern::loss(const ArrayDouble &coeffs) {
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

  ArrayDouble local_coeffs(1 + n_nodes);
  local_coeffs[0] = coeffs[i];
  view(local_coeffs, 1).mult_fill(view(coeffs, n_nodes + i * n_nodes, n_nodes + (i + 1) *n_nodes), 1);
  return model_ptr->loss(local_coeffs);
}

double HawkesSDCALoglikKern::get_decay() const {
  return decay;
}

void HawkesSDCALoglikKern::set_decay(const double decay) {
  if (decay <= 0) {
    TICK_ERROR("decay must be positive, received " << decay);
  }
  this->decay = decay;
  weights_computed = false;
}

SArrayDoublePtr HawkesSDCALoglikKern::get_iterate()  {

  if (sdca_list.size() != n_nodes){
    SArrayDoublePtr zero_iterate = SArrayDouble::new_ptr((1 + n_nodes) * n_nodes);
    zero_iterate->init_to_zero();
    return zero_iterate;
  }

  ulong n_coeffs_per_subproblem = G[0].size();
  ArrayDouble iterate(n_nodes * n_coeffs_per_subproblem);
  for (ulong i = 0; i < n_nodes; ++i) {
    ArrayDouble sdca_iterate(n_coeffs_per_subproblem);
    sdca_list[i].get_iterate(sdca_iterate);
    iterate[i] = sdca_iterate[0];
    for (ulong j = 0; j < n_nodes; ++j) {
      iterate[n_nodes + n_nodes * i + j] = sdca_iterate[1 + j];
    }
  }
  return iterate.as_sarray_ptr();
}
