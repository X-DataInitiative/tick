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

  std::cout << "before" << std::endl;
  (*G_buffer)[0].print();
//  parallel_run(get_n_threads(), n_nodes * n_realizations,
//               &HawkesSDCALoglikKern::compute_weights_dim_i, this, G_buffer);

  for (ulong i_r = 0; i_r < n_nodes * n_realizations; ++i_r) {
//    compute_weights_dim_i(i_r, G_buffer);
    const auto r = static_cast<const ulong>(i_r / n_nodes);
    const ulong i = i_r % n_nodes;

    const ArrayDouble t_i = view(*timestamps_list[r][i]);
    ArrayDouble2d g_i = view(g[i]);
//  ArrayDouble G_i_r = view_row((*G_buffer)[i], r);
//  G_i_r.print();

    const ulong n_jumps_i = (*n_jumps_per_node)[i];
    const ulong start_index = view(*get_n_jumps_per_realization(), 0, r).sum();

    auto get_index_g = [=](ulong k, ulong j) {
      return 1 + start_index + (1 + n_nodes) * k + j;
    };

    for (ulong j = 0; j < n_nodes; j++) {
      for (ulong k = 0; k < n_jumps_i + 1; k++) {
        std::cout << "1 + start_index + (1 + n_nodes) * k = " << 1 + start_index + (1 + n_nodes) * k
                                                              << ", size=" << g_i.size() << std::endl;
        g_i[1 + start_index + (1 + n_nodes) * k] = 1.;
      }
    }
  }

  std::cout << "will print" << std::endl;
  G[0].print();
  (*G_buffer)[0].print();

  for (ulong i = 0; i < n_nodes; ++i) {
    for (ulong r = 0; r < n_realizations; ++r) {
      (*G_buffer)[i].print();
      G[i].print();
      G[i].mult_incr(view_row((*G_buffer)[i], r), 1);
    }
    G[i][0] = (*end_times)[i];
  }

  for (ulong i = 0; i < n_nodes; ++i) {
    auto model_i_ptr = std::make_shared<ModelHawkesSDCAOneNode>(g[i], G[i]);

    const ulong epoch_size = g[i].n_rows();
    sdca_list.emplace_back(SDCA(l_l2sq, epoch_size, tol, rand_type, seed));
    sdca_list[i].set_model(model_i_ptr);
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
//  ArrayDouble G_i_r = view_row((*G_buffer)[i], r);
//  G_i_r.print();

  const ulong n_jumps_i = (*n_jumps_per_node)[i];
  const ulong start_index = view(*get_n_jumps_per_realization(), 0, r).sum();

  auto get_index_g = [=](ulong k, ulong j) {
    return 1 + start_index + (1 + n_nodes) * k + j;
  };

  for (ulong j = 0; j < n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps_list[r][j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const double t_i_k = k < n_jumps_i ? t_i[k] : (*end_times)[r];
      if (k > 0) {
        const double ebt = std::exp(-decay * (t_i_k - t_i[k - 1]));

        if (k < n_jumps_i) g_i[get_index_g(k, j)] = g_i[get_index_g(k - 1, j)] * ebt;
//        G_i_r[1 + j] += g_i[get_index_g(k - 1, j)] * (1 - ebt) / decay;
      } else {
        g_i[get_index_g(k, j)] = 0;
//        G_i_r[1 + j] = 0;
      }

      while ((ij < (*n_jumps_per_node)[j]) && (t_j[ij] < t_i_k)) {
        const double ebt = std::exp(-decay * (t_i_k - t_j[ij]));
        if (k < n_jumps_i) g_i[get_index_g(k, j)] += decay * ebt;
//        G_i_r[1 + j] += 1 - ebt;
        ij++;
      }
      g_i[get_index_g(k, 0)] = 1.;
    }
  }
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
