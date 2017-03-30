
#include "hawkes_fixed_sumexpkern_leastsq.h"

ModelHawkesFixedSumExpKernLeastSq::ModelHawkesFixedSumExpKernLeastSq(
    const ArrayDouble &decays,
    const unsigned int max_n_threads,
    const unsigned int optimization_level)
    : ModelHawkesSingle(max_n_threads, optimization_level),
      decays(decays), n_decays(decays.size()) {}

// Method that computes the value
double ModelHawkesFixedSumExpKernLeastSq::loss(const ArrayDouble &coeffs) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of the contribution of each component
  SArrayDoublePtr values =
      parallel_map(get_n_threads(),
                   n_nodes,
                   &ModelHawkesFixedSumExpKernLeastSq::loss_i,
                   this,
                   coeffs);

  // We just need to sum up the contribution
  return values->sum() / n_total_jumps;
}

// Performs the computation of the contribution of the i component to the value
double ModelHawkesFixedSumExpKernLeastSq::loss_i(const ulong i,
                                                 const ArrayDouble &coeffs) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

  double mu_i = coeffs[i];
  ulong start_alpha_i = n_nodes + i * n_nodes * n_decays;
  ulong end_alpha_i = n_nodes + (i + 1) * n_nodes * n_decays;
  ArrayDouble alpha_i = view(coeffs, start_alpha_i, end_alpha_i);

  double C_sum = 0;
  double Dg_sum = 0;
  double Dgg_sum = 0;
  double E_sum = 0;

  ArrayDouble2d &C_i = C[i];
  for (ulong j = 0; j < n_nodes; ++j) {
    ArrayDouble &Dg_j = Dg[j];
    ArrayDouble2d &Dgg_j = Dgg[j];
    ArrayDouble2d &E_j = E[j];

    for (ulong u = 0; u < n_decays; ++u) {
      double alpha_i_j_u = alpha_i[j * n_decays + u];
      C_sum += alpha_i_j_u * view_row(C_i, j)[u];
      Dg_sum += alpha_i_j_u * Dg_j[u];

      for (ulong u1 = 0; u1 < n_decays; ++u1) {
        double alpha_i_j_u1 = alpha_i[j * n_decays + u1];
        Dgg_sum += alpha_i_j_u * alpha_i_j_u1 * view_row(Dgg_j, u)[u1];

        for (ulong j1 = 0; j1 < n_nodes; ++j1) {
          double alpha_i_j1_u1 = alpha_i[j1 * n_decays + u1];
          E_sum += alpha_i_j_u * alpha_i_j1_u1 * view_row(E_j, j1)[u * n_decays + u1];
        }
      }
    }
  }

  double A_i = mu_i * mu_i * end_time;
  A_i += 2 * mu_i * Dg_sum;
  A_i += Dgg_sum;
  A_i += 2 * E_sum;

  double B_i = mu_i * (*n_jumps_per_node)[i];
  B_i += C_sum;

  return A_i - 2 * B_i;
}

// Method that computes the gradient
void ModelHawkesFixedSumExpKernLeastSq::grad(const ArrayDouble &coeffs,
                                             ArrayDouble &out) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesFixedSumExpKernLeastSq::grad_i,
               this,
               coeffs,
               out);
  out /= n_total_jumps;
}

// Method that computes the component i of the gradient
void ModelHawkesFixedSumExpKernLeastSq::grad_i(const ulong i,
                                               const ArrayDouble &coeffs,
                                               ArrayDouble &out) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

  double mu_i = coeffs[i];
  ulong start_alpha_i = n_nodes + i * n_nodes * n_decays;
  ulong end_alpha_i = n_nodes + (i + 1) * n_nodes * n_decays;
  ArrayDouble alpha_i = view(coeffs, start_alpha_i, end_alpha_i);

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, start_alpha_i, end_alpha_i);
  grad_alpha_i.init_to_zero();

  grad_mu_i = 2 * mu_i * end_time - 2 * (*n_jumps_per_node)[i];

  ArrayDouble2d &C_i = C[i];
  for (ulong j = 0; j < n_nodes; ++j) {
    ArrayDouble &Dg_j = Dg[j];
    ArrayDouble2d &Dgg_j = Dgg[j];
    ArrayDouble2d &E_j = E[j];

    for (ulong u = 0; u < n_decays; ++u) {
      double alpha_i_j_u = alpha_i[j * n_decays + u];
      double &grad_alpha_i_j_u = grad_alpha_i[j * n_decays + u];

      grad_alpha_i_j_u -= 2 * view_row(C_i, j)[u];
      grad_mu_i += 2 * alpha_i_j_u * Dg_j[u];
      grad_alpha_i_j_u += 2 * mu_i * Dg_j[u];

      for (ulong u1 = 0; u1 < n_decays; ++u1) {
        double alpha_i_j_u1 = alpha_i[j * n_decays + u1];

        grad_alpha_i_j_u += 2 * alpha_i_j_u1 * view_row(Dgg_j, u)[u1];

        for (ulong j1 = 0; j1 < n_nodes; ++j1) {
          double alpha_i_j1_u1 = alpha_i[j1 * n_decays + u1];
          double &grad_alpha_i_j1_u1 = grad_alpha_i[j1 * n_decays + u1];
          double E_j_j1_u_u1 = view_row(E_j, j1)[u * n_decays + u1];

          grad_alpha_i_j_u += 2 * alpha_i_j1_u1 * E_j_j1_u_u1;
          grad_alpha_i_j1_u1 += 2 * alpha_i_j_u * E_j_j1_u_u1;
        }
      }
    }
  }
}

// Computes both gradient and value
// TODO : optimization !
double ModelHawkesFixedSumExpKernLeastSq::loss_and_grad(const ArrayDouble &coeffs,
                                                        ArrayDouble &out) {
  grad(coeffs, out);
  return loss(coeffs);
}

// Contribution of the ith component to the initialization
// Computation of the arrays H, Dg, Dg2 and C
void ModelHawkesFixedSumExpKernLeastSq::compute_weights_i(const ulong i) {
  ulong n_decays = decays.size();

  ArrayDouble &timestamps_i = *timestamps[i];
  ArrayDouble2d H(n_nodes, n_decays);
  H.init_to_zero();
  ArrayULong l = ArrayULong(n_nodes);
  l.init_to_zero();

  ArrayDouble2d &C_i = C[i];
  ArrayDouble &Dg_i = Dg[i];
  ArrayDouble2d &Dgg_i = Dgg[i];
  ArrayDouble2d &E_i = E[i];

  ulong N_i = timestamps_i.size();
  for (ulong k = 0; k < N_i; ++k) {
    double t_k_i = timestamps_i[k];

    for (ulong j = 0; j < n_nodes; ++j) {
      ArrayDouble &timestamps_j = *timestamps[j];
      ulong N_j = timestamps_j.size();

      if (k > 0) {
        double t_k_minus_one_i = timestamps_i[k - 1];

        for (ulong u = 0; u < n_decays; ++u) {
          double decay_u = decays[u];
          view_row(H, j)[u] *= cexp(-decay_u * (t_k_i - t_k_minus_one_i));
        }
      }

      while (l[j] < N_j && timestamps_j[l[j]] < t_k_i) {
        double t_l_j = timestamps_j[l[j]];

        for (ulong u = 0; u < n_decays; ++u) {
          double decay_u = decays[u];
          view_row(H, j)[u] += decay_u * cexp(-decay_u * (t_k_i - t_l_j));
        }

        l[j] += 1;
      }

      for (ulong u = 0; u < n_decays; ++u) {
        double decay_u = decays[u];
        view_row(C_i, j)[u] += view_row(H, j)[u];

        for (ulong u1 = 0; u1 < n_decays; ++u1) {
          double decay_u1 = decays[u1];

          // we fill E_i,j,u',u
          double ratio = decay_u1 / (decay_u1 + decay_u);
          double tmp = 1 - cexp(-(decay_u1 + decay_u) * (end_time - t_k_i));
          view_row(E_i, j)[u1 * n_decays + u] += ratio * tmp * view_row(H, j)[u];
        }
      }
    }

    for (ulong u = 0; u < n_decays; ++u) {
      double decay_u = decays[u];
      Dg_i[u] += 1 - cexp(-decay_u * (end_time - t_k_i));

      for (ulong u1 = 0; u1 < n_decays; ++u1) {
        double decay_u1 = decays[u1];

        double ratio = decay_u * decay_u1 / (decay_u + decay_u1);
        view_row(Dgg_i, u)[u1] += ratio * (1 - cexp(-(decay_u + decay_u1) * (end_time - t_k_i)));
      }
    }
  }
}

void ModelHawkesFixedSumExpKernLeastSq::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }

  C = ArrayDouble2dList1D(n_nodes);
  Dg = ArrayDoubleList1D(n_nodes);
  Dgg = ArrayDouble2dList1D(n_nodes);
  E = ArrayDouble2dList1D(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    C[i] = ArrayDouble2d(n_nodes, n_decays);
    C[i].init_to_zero();
    Dg[i] = ArrayDouble(n_decays);
    Dg[i].init_to_zero();
    Dgg[i] = ArrayDouble2d(n_decays, n_decays);
    Dgg[i].init_to_zero();
    E[i] = ArrayDouble2d(n_nodes, n_decays * n_decays);
    E[i].init_to_zero();
  }
}

// Full initialization of the arrays H, Dg, Dg2 and C
// Must be performed just once
void ModelHawkesFixedSumExpKernLeastSq::compute_weights() {
  allocate_weights();

  // Multithreaded computation of the arrays
  parallel_run(get_n_threads(), n_nodes,
               &ModelHawkesFixedSumExpKernLeastSq::compute_weights_i,
               this);
  weights_computed = true;
}

ulong ModelHawkesFixedSumExpKernLeastSq::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * n_decays;
}
