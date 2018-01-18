// License: BSD 3 clause


#include "tick/hawkes/model/model_hawkes_expkern_loglik_single.h"

ModelHawkesExpKernLogLikSingle::ModelHawkesExpKernLogLikSingle(
  const double decay, const int max_n_threads) :
  ModelHawkesLogLikSingle(max_n_threads),
  decay(decay) {}

void ModelHawkesExpKernLogLikSingle::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDouble2dList1D(n_nodes);
  sum_G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    g[i] = ArrayDouble2d((*n_jumps_per_node)[i], n_nodes);
    g[i].init_to_zero();
    G[i] = ArrayDouble2d((*n_jumps_per_node)[i] + 1, n_nodes);
    G[i].init_to_zero();
    sum_G[i] = ArrayDouble(n_nodes);
  }
}

void ModelHawkesExpKernLogLikSingle::compute_weights_dim_i(const ulong i) {
  const ArrayDouble t_i = view(*timestamps[i]);
  ArrayDouble2d g_i = view(g[i]);
  ArrayDouble2d G_i = view(G[i]);
  ArrayDouble sum_G_i = view(sum_G[i]);

  const ulong n_jumps_i = (*n_jumps_per_node)[i];

  for (ulong j = 0; j < n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps[j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const double t_i_k = k < n_jumps_i ? t_i[k] : end_time;
      if (k > 0) {
        const double ebt = std::exp(-decay * (t_i_k - t_i[k - 1]));

        if (k < n_jumps_i) g_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * ebt;
        G_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * (1 - ebt) / decay;
      } else {
        if (k < n_jumps_i) g_i[k * n_nodes + j] = 0;
        G_i[k * n_nodes + j] = 0;
        sum_G[i][j] = 0.;
      }

      while ((ij < (*n_jumps_per_node)[j]) && (t_j[ij] < t_i_k)) {
        const double ebt = std::exp(-decay * (t_i_k - t_j[ij]));
        if (k < n_jumps_i) g_i[k * n_nodes + j] += decay * ebt;
        G_i[k * n_nodes + j] += 1 - ebt;
        ij++;
      }
      sum_G_i[j] += G_i[k * n_nodes + j];
    }
  }
}

ulong ModelHawkesExpKernLogLikSingle::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
