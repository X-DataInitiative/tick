// License: BSD 3 clause


#include "hawkes_fixed_expkern_loglik.h"
#include "hawkes_sdca_loglik_kern.h"


HawkesSDCALoglikKern::HawkesSDCALoglikKern(double decay, double l_l2sq,
                                           int max_n_threads, double tol,
                                           RandType rand_type, int seed)
  : ModelHawkesList(max_n_threads, optimization_level),
    weights_allocated(false), l_l2sq(l_l2sq), tol(tol), rand_type(rand_type), seed(seed) {
  set_decay(decay);
}

void HawkesSDCALoglikKern::compute_weights() {
  if (!weights_allocated) allocate_weights();

  auto G_buffer = std::make_shared<ArrayDouble2dList1D>(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      std::cout << "initialize" << i << " " << r << std::endl;
      (*G_buffer)[i] = ArrayDouble2d(n_realizations, 1ul + n_nodes, nullptr);
      (*G_buffer)[i].init_to_zero();
    }
  }

  for (ulong i = 0; i < n_nodes; i++) {
    g[i].init_to_zero();
    G[i].init_to_zero();
  }

//  parallel_run(get_n_threads(), n_nodes * n_realizations,
//               &HawkesSDCALoglikKern::compute_weights_dim_i, this, G_buffer);

  for (ulong i_r = 0; i_r < n_nodes * n_realizations; ++i_r) {
    compute_weights_dim_i(i_r, G_buffer);
  }

  for (int i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      G[i].mult_incr(view_row((*G_buffer)[i], r), 1);
    }
  }

  for (ulong i = 0; i < n_nodes; i++) {
    g[i].print();
    G[i].print();
  }

  weights_computed = true;
}

void HawkesSDCALoglikKern::allocate_weights(){
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    ulong n_jumps_node_i = (*n_jumps_per_node)[i];
    g[i] = ArrayDouble2d(n_jumps_node_i, 1 + n_nodes, nullptr);
    G[i] = ArrayDouble(1 + n_nodes);
  }

  weights_allocated = true;
}

void HawkesSDCALoglikKern::compute_weights_dim_i(const ulong i_r, std::shared_ptr<ArrayDouble2dList1D> G_buffer) {
  const auto r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;

  const ArrayDouble t_i = view(*timestamps_list[r][i]);
  ArrayDouble2d g_i = view(g[i]);
  ArrayDouble G_i_r = view_row((*G_buffer)[i], r);

  const double end_time = (*end_times)[r];
  const ulong n_jumps_i = t_i.size();
  ulong start_row = 0;
  for (int smaller_r = 0; smaller_r < r; ++smaller_r) {
    start_row += timestamps_list[r][i]->size();
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
//  g_i.print();
//  (*G_buffer)[i].print();
}

// The main method for performing one iteration
void HawkesSDCALoglikKern::solve(ArrayDouble &mu, ArrayDouble2d &adjacency,
                                 ArrayDouble2d &z1, ArrayDouble2d &z2,
                                 ArrayDouble2d &u1, ArrayDouble2d &u2) {
  if (!weights_computed) compute_weights();


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
