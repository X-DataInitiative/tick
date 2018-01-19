// License: BSD 3 clause


#include "tick/hawkes/model/model_hawkes_sumexpkern_loglik_single.h"

ModelHawkesSumExpKernLogLikSingle::ModelHawkesSumExpKernLogLikSingle() :
  ModelHawkesLogLikSingle(), decays(0) {}

ModelHawkesSumExpKernLogLikSingle::ModelHawkesSumExpKernLogLikSingle(
  const ArrayDouble &decays, const int max_n_threads) :
  ModelHawkesLogLikSingle(max_n_threads),
  decays(decays) {}

void ModelHawkesSumExpKernLogLikSingle::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDouble2dList1D(n_nodes);
  sum_G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    g[i] = ArrayDouble2d((*n_jumps_per_node)[i], n_nodes * get_n_decays());
    g[i].init_to_zero();
    G[i] = ArrayDouble2d((*n_jumps_per_node)[i] + 1, n_nodes * get_n_decays());
    G[i].init_to_zero();
    sum_G[i] = ArrayDouble(n_nodes * get_n_decays());
  }
}

void ModelHawkesSumExpKernLogLikSingle::compute_weights_dim_i(const ulong i) {
  const ArrayDouble t_i = view(*timestamps[i]);
  ArrayDouble2d g_i = view(g[i]);
  ArrayDouble2d G_i = view(G[i]);
  ArrayDouble sum_G_i = view(sum_G[i]);

  const ulong n_jumps_i = (*n_jumps_per_node)[i];

  auto get_index = [=](ulong k, ulong j, ulong u) {
    return n_nodes * get_n_decays() * k + get_n_decays() * j + u;
  };

  for (ulong j = 0; j < n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps[j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const double t_i_k = k < n_jumps_i ? t_i[k] : end_time;

      for (ulong u = 0; u < get_n_decays(); ++u) {
        if (k > 0) {
          const double ebt = std::exp(-decays[u] * (t_i_k - t_i[k - 1]));

          if (k < n_jumps_i) g_i[get_index(k, j, u)] = g_i[get_index(k - 1, j, u)] * ebt;
          G_i[get_index(k, j, u)] = g_i[get_index(k - 1, j, u)] * (1 - ebt) / decays[u];

        } else {
          if (k < n_jumps_i) g_i[get_index(k, j, u)] = 0;
          G_i[get_index(k, j, u)] = 0;
          sum_G_i[j * get_n_decays() + u] = 0.;
        }
      }

      while ((ij < (*n_jumps_per_node)[j]) && (t_j[ij] < t_i_k)) {
        for (ulong u = 0; u < get_n_decays(); ++u) {
          const double ebt = std::exp(-decays[u] * (t_i_k - t_j[ij]));
          if (k < n_jumps_i) g_i[get_index(k, j, u)] += decays[u] * ebt;
          G_i[get_index(k, j, u)] += 1 - ebt;
        }
        ij++;
      }
      for (ulong u = 0; u < get_n_decays(); ++u) {
        sum_G_i[j * get_n_decays() + u] += G_i[get_index(k, j, u)];
      }
    }
  }
}

ulong ModelHawkesSumExpKernLogLikSingle::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays();
}
