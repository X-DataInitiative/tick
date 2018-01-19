// License: BSD 3 clause


#include "tick/base/base.h"
#include "tick/hawkes/inference/hawkes_adm4.h"

HawkesADM4::HawkesADM4(const double decay, const double rho,
                       const int max_n_threads, const unsigned int optimization_level)
  : ModelHawkesList(max_n_threads, optimization_level) {
  set_decay(decay);
  set_rho(rho);
}

void HawkesADM4::compute_weights() {
  // Allocate weights
  next_mu = ArrayDouble2d(n_realizations, n_nodes);
  next_C = ArrayDouble2d(n_realizations * n_nodes, n_nodes);
  unnormalized_next_C = ArrayDouble2d(n_realizations * n_nodes, n_nodes);

  kernel_integral = ArrayDouble(n_nodes);
  g = ArrayDouble2dList2D(n_realizations);
  for (ulong r = 0; r < n_realizations; ++r) {
    g[r] = ArrayDouble2dList1D(n_nodes);
    for (ulong u = 0; u < n_nodes; ++u) {
      g[r][u] = ArrayDouble2d(timestamps_list[r][u]->size(), n_nodes);
    }
  }

  // Compute weights
  // variable to compute kernel integral in parallel that will be reduced afterwards
  ArrayDouble2d map_kernel_integral(n_realizations, n_nodes);
  map_kernel_integral.init_to_zero();
  parallel_run(get_n_threads(), n_nodes * n_realizations, &HawkesADM4::compute_weights_ru, this,
               map_kernel_integral);

  kernel_integral.init_to_zero();
  for (ulong r = 0; r < n_realizations; ++r) {
    ArrayDouble map_kernel_integral_r = view_row(map_kernel_integral, r);
    for (ulong u = 0; u < n_nodes; ++u) {
      kernel_integral[u] += map_kernel_integral_r[u];
    }
  }

  weights_computed = true;
}

// This code is very similar to ModelHawkesExpKernLogLikSingle
void HawkesADM4::compute_weights_ru(const ulong r_u, ArrayDouble2d &map_kernel_integral) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong u = r_u % n_nodes;
  ArrayDouble2d g_ru = view(g[r][u]);
  const ArrayDouble timestamps_ru = view(*timestamps_list[r][u]);
  const double end_time_r = (*end_times)[r];
  ArrayDouble map_kernel_integral_r = view_row(map_kernel_integral, r);

  for (ulong v = 0; v < n_nodes; v++) {
    const ArrayDouble timestamps_rv = view(*timestamps_list[r][v]);
    ulong ij = 0;
    for (ulong k = 0; k < timestamps_ru.size(); k++) {
      const double t_ru_k = timestamps_ru[k];

      if (k > 0) {
        const double ebt = cexp(-decay * (t_ru_k - timestamps_ru[k - 1]));
        g_ru[k * n_nodes + v] = g_ru[(k - 1) * n_nodes + v] * ebt;
      } else {
        if (k < timestamps_ru.size()) g_ru[k * n_nodes + v] = 0;
      }
      while ((ij < timestamps_rv.size()) && (timestamps_rv[ij] < t_ru_k)) {
        const double ebt = cexp(-decay * (t_ru_k - timestamps_rv[ij]));
        g_ru[k * n_nodes + v] += decay * ebt;
        ij++;
      }

      if (u == v) {
        // We use this pass over the data to fill kernel_integral
        map_kernel_integral_r[u] += (1. - cexp(-decay * (end_time_r - t_ru_k)));
      }
    }
  }
}

// The main method for performing one iteration
void HawkesADM4::solve(ArrayDouble &mu, ArrayDouble2d &adjacency,
                       ArrayDouble2d &z1, ArrayDouble2d &z2,
                       ArrayDouble2d &u1, ArrayDouble2d &u2) {
  if (!weights_computed) compute_weights();

  if (mu.size() != n_nodes) {
    TICK_ERROR("mu argument must be an array of shape (" << n_nodes << ",)");
  }
  if (adjacency.n_rows() != n_nodes || adjacency.n_cols() != n_nodes) {
    TICK_ERROR("adjacency matrix must be an array of shape (" << n_nodes << ", " << n_nodes << ")");
  }
  if (z1.n_rows() != n_nodes || z1.n_cols() != n_nodes) {
    TICK_ERROR("Z1 matrix must be an array of shape (" << n_nodes << ", " << n_nodes << ")");
  }
  if (z2.n_rows() != n_nodes || z2.n_cols() != n_nodes) {
    TICK_ERROR("Z2 matrix must be an array of shape (" << n_nodes << ", " << n_nodes << ")");
  }
  if (u1.n_rows() != n_nodes || u1.n_cols() != n_nodes) {
    TICK_ERROR("U1 matrix must be an array of shape (" << n_nodes << ", " << n_nodes << ")");
  }
  if (u2.n_rows() != n_nodes || u2.n_cols() != n_nodes) {
    TICK_ERROR("U2 matrix must be an array of shape (" << n_nodes << ", " << n_nodes << ")");
  }

  next_C.init_to_zero();
  next_mu.init_to_zero();

  parallel_run(get_n_threads(), n_nodes * n_realizations,
               &HawkesADM4::estimate_ru, this, mu, adjacency);
  parallel_run(std::min(get_n_threads(), static_cast<const unsigned int>(n_nodes)), n_nodes,
               &HawkesADM4::update_u, this, mu, adjacency, z1, z2, u1, u2);
}

// Procedure called by HawkesADM4::solve
void HawkesADM4::estimate_ru(const ulong r_u,
                             ArrayDouble &mu, ArrayDouble2d &adjacency) {
  // Obtain realization and node index from r_u
  const ulong r = static_cast<const ulong>(r_u / n_nodes);
  const ulong node_u = r_u % n_nodes;

  // Fetch corresponding data
  SArrayDoublePtrList1D &realization = timestamps_list[r];
  ArrayDouble adjacency_u = view_row(adjacency, node_u);
  ArrayDouble2d g_ru = view(g[r][node_u]);
  double mu_u = mu[node_u];

  // initialize next data
  double &next_mu_ur = next_mu(r, node_u);
  ArrayDouble next_C_ru = view_row(next_C, r * n_nodes + node_u);
  ArrayDouble unnormalized_next_C_ru = view_row(unnormalized_next_C, r * n_nodes + node_u);

  // We loop in reverse order to benefit from last_indices
  for (ulong i = realization[node_u]->size() - 1; i != static_cast<ulong>(-1); i--) {
    // this array will store temporary values
    unnormalized_next_C_ru.init_to_zero();
    ArrayDouble g_ru_i = view_row(g_ru, i);

    // norm will be equal to mu_u + \sum_v \sum_(t_j < t_i) a_uv g(t_i - t_j)
    double norm = mu_u;

    for (ulong node_v = 0; node_v < n_nodes; node_v++) {
      const double sum_unnormalized_p_ij = adjacency_u[node_v] * g_ru_i[node_v];
      unnormalized_next_C_ru[node_v] += sum_unnormalized_p_ij;
      norm += sum_unnormalized_p_ij;
    }

    next_mu_ur += mu_u / norm;
    next_C_ru.mult_incr(unnormalized_next_C_ru, 1. / norm);
  }
}

// A method called in parallel by the method 'solve' (see below)
void HawkesADM4::update_u(const ulong u, ArrayDouble &mu, ArrayDouble2d &adjacency,
                          ArrayDouble2d &z1, ArrayDouble2d &z2,
                          ArrayDouble2d &u1, ArrayDouble2d &u2) {
  ArrayDouble adjacency_u = view_row(adjacency, u);

  ArrayDouble z1_u = view_row(z1, u);
  ArrayDouble z2_u = view_row(z2, u);
  ArrayDouble u1_u = view_row(u1, u);
  ArrayDouble u2_u = view_row(u2, u);

  update_adjacency_u(u, adjacency_u, z1_u, z2_u, u1_u, u2_u);
  update_baseline_u(u, mu);
}

// Procedure called by HawkesADM4::solve
void HawkesADM4::update_adjacency_u(const ulong u, ArrayDouble &adjacency_u,
                                    ArrayDouble &z1_u, ArrayDouble &z2_u,
                                    ArrayDouble &u1_u, ArrayDouble &u2_u) {
  for (ulong v = 0; v < n_nodes; v++) {
    const double B = kernel_integral[v] + rho * (-z1_u[v] + u1_u[v] - z2_u[v] + u2_u[v]);

    double C = 0;
    for (ulong r = 0; r < n_realizations; ++r) {
      C += next_C(r * n_nodes + u, v);
    }

    // computation of updated value
    adjacency_u[v] = (-B + sqrt(B * B + 8 * rho * C)) / (4 * rho);
  }
}

void HawkesADM4::update_baseline_u(const ulong u, ArrayDouble &mu) {
  mu[u] = 0;
  for (ulong r = 0; r < n_realizations; ++r) {
    mu[u] += next_mu(r, u) / end_times->sum();
  }
}

double HawkesADM4::get_decay() const {
  return decay;
}

void HawkesADM4::set_decay(const double decay) {
  if (decay <= 0) {
    TICK_ERROR("decay must be positive, received " << decay);
  }
  this->decay = decay;
  weights_computed = false;
}

double HawkesADM4::get_rho() const {
  return rho;
}

void HawkesADM4::set_rho(double rho) {
  if (rho <= 0) {
    TICK_ERROR("rho (penalty parameter) must be positive, received " << rho);
  }
  this->rho = rho;
}
