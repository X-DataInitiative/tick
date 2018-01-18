// License: BSD 3 clause


#include "tick/hawkes/model/model_hawkes_expkern_leastsq_single.h"

// Constructor
ModelHawkesExpKernLeastSqSingle::ModelHawkesExpKernLeastSqSingle(
  const SArrayDouble2dPtr decays,
  const int max_n_threads,
  const unsigned int optimization_level)
  : ModelHawkesSingle(max_n_threads, optimization_level), decays(decays) {}

// Method that computes the value
double ModelHawkesExpKernLeastSqSingle::loss(const ArrayDouble &coeffs) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of the
  // contribution of each component
  const double loss_sum =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesExpKernLeastSqSingle::loss_i,
                                 this, coeffs);

  // We just need to sum up the contribution
  return loss_sum / n_total_jumps;
}

// Performs the computation of the contribution of the i component to the value
double ModelHawkesExpKernLeastSqSingle::loss_i(const ulong i, const ArrayDouble &coeffs) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling loss_i");

  const ArrayDouble E_i = view_row(E, i);
  const ArrayDouble Dg_i = view_row(Dg, i);
  const ArrayDouble Dg2_i = view_row(Dg2, i);
  const ArrayDouble C_i = view_row(C, i);
  const ArrayDouble mu = view(coeffs, 0, n_nodes);
  const ArrayDouble alpha = view(coeffs, n_nodes, n_nodes + n_nodes * n_nodes);

  double value = 0;
  value += mu[i] * mu[i] * end_time;

  double temp1 = 0;
  double temp2 = 0;
  double temp3 = 0;
  double temp4 = 0;
  for (ulong j = 0; j < n_nodes; j++) {
    temp1 += alpha[i * n_nodes + j] * Dg_i[j];
    temp2 += alpha[i * n_nodes + j] * alpha[i * n_nodes + j] * Dg2_i[j];
    temp3 += alpha[i * n_nodes + j] * C_i[j];
    for (ulong j1 = 0; j1 < n_nodes; j1++) {
      temp4 += alpha[i * n_nodes + j] * alpha[i * n_nodes + j1] *
        E_i[j * n_nodes + j1];
    }
  }
  value += 2 * mu[i] * temp1 + temp2 - 2 * temp3 + 2 * temp4 -
    2 * mu[i] * (*n_jumps_per_node)[i];
  return value;
}

// Method that computes the gradient
void ModelHawkesExpKernLeastSqSingle::grad(const ArrayDouble &coeffs, ArrayDouble &out) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesExpKernLeastSqSingle::grad_i,
               this,
               coeffs,
               out);
  out /= n_total_jumps;
}

// Method that computes the component i of the gradient
void ModelHawkesExpKernLeastSqSingle::grad_i(const ulong i, const ArrayDouble &coeffs,
                                             ArrayDouble &out) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling grad_i");

  const ArrayDouble E_i = view_row(E, i);
  const ArrayDouble Dg_i = view_row(Dg, i);
  const ArrayDouble Dg2_i = view_row(Dg2, i);
  const ArrayDouble C_i = view_row(C, i);

  const ArrayDouble mu = view(coeffs, 0, n_nodes);
  const ArrayDouble alpha = view(coeffs, n_nodes, n_nodes + n_nodes * n_nodes);
  ArrayDouble grad_mu = view(out, 0, n_nodes);
  ArrayDouble grad_alpha = view(out, n_nodes, n_nodes + n_nodes * n_nodes);

  grad_mu[i] = 2 * mu[i] * end_time - 2 * (*n_jumps_per_node)[i];

  for (ulong j = 0; j < n_nodes; j++) {
    grad_mu[i] += 2 * alpha[i * n_nodes + j] * Dg_i[j];
    grad_alpha[i * n_nodes + j] =
      2 * mu[i] * Dg_i[j] + 2 * alpha[i * n_nodes + j] * Dg2_i[j] +
        4 * alpha[i * n_nodes + j] * E_i[j * n_nodes + j] - 2 * C_i[j];

    for (ulong j1 = 0; j1 < n_nodes; j1++) {
      if (j1 != j)
        grad_alpha[i * n_nodes + j] += 2 * alpha[i * n_nodes + j1] *
          (E_i[j * n_nodes + j1] +
            E_i[j1 * n_nodes + j]);
    }
  }
}

void ModelHawkesExpKernLeastSqSingle::hessian(ArrayDouble &out) {
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(), n_nodes, &ModelHawkesExpKernLeastSqSingle::hessian_i, this, out);
  out /= n_total_jumps;
}

void ModelHawkesExpKernLeastSqSingle::hessian_i(const ulong i, ArrayDouble &out) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

  // fill mu line of matrix
  const ulong start_mu_line = i * (n_nodes + 1);
  // fill mu in mu diag
  out[start_mu_line] = 2 * end_time;
  // fill alpha line
  const ArrayDouble Dg_i = view_row(Dg, i);
  for (ulong j = 0; j < n_nodes; ++j) {
    out[start_mu_line + j + 1] += 2 * Dg_i[j];
  }

  // fill alpha lines
  const ArrayDouble E_k = view_row(E, i);
  const ArrayDouble Dg2_k = view_row(Dg2, i);
  const ArrayDouble Dg_k = view_row(Dg, i);

  const ulong block_start = (i + 1) * n_nodes * (n_nodes + 1);
  for (ulong l = 0; l < n_nodes; ++l) {
    const ulong start_alpha_line = block_start + l * (n_nodes + 1);
    out[start_alpha_line] += 2 * Dg_k[l];
    for (ulong m = 0; m < n_nodes; ++m) {
      out[start_alpha_line + m + 1] += 2 * (E_k[l * n_nodes + m] + E_k[m * n_nodes + l]);
      if (l == m) {
        out[start_alpha_line + m + 1] += 2 * Dg2_k[l];
      }
    }
  }
}

// Computes both gradient and value
// TODO : optimization !
double ModelHawkesExpKernLeastSqSingle::loss_and_grad(const ArrayDouble &coeffs,
                                                      ArrayDouble &out) {
  grad(coeffs, out);
  return loss(coeffs);
}

void ModelHawkesExpKernLeastSqSingle::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }

  // Allocation
  Dg = ArrayDouble2d(n_nodes, n_nodes);
  Dg.init_to_zero();
  Dg2 = ArrayDouble2d(n_nodes, n_nodes);
  Dg2.init_to_zero();
  C = ArrayDouble2d(n_nodes, n_nodes);
  C.init_to_zero();
  E = ArrayDouble2d(n_nodes, n_nodes * n_nodes);
  E.init_to_zero();
}

// Full initialization of the arrays H, Dg, Dg2 and C
// Must be performed just once
void ModelHawkesExpKernLeastSqSingle::compute_weights() {
  allocate_weights();
  parallel_run(get_n_threads(), n_nodes, &ModelHawkesExpKernLeastSqSingle::compute_weights_i, this);
  weights_computed = true;
}

// Contribution of the ith component to the initialization
// Computation of the arrays H, Dg, Dg2 and C
void ModelHawkesExpKernLeastSqSingle::compute_weights_i(const ulong i) {
  const SArrayDoublePtr timestamps_i = timestamps[i];
  ArrayDouble2d H(n_nodes, n_nodes);
  H.init_to_zero();
  ArrayDouble Dg_i = view_row(Dg, i);
  ArrayDouble Dg2_i = view_row(Dg2, i);
  ArrayDouble C_i = view_row(C, i);

  const ulong N_i_size = timestamps_i->size();
  for (ulong j = 0; j < n_nodes; j++) {
    const SArrayDoublePtr realization_j = timestamps[j];
    const ulong N_j_size = realization_j->size();
    const double betaij = (*decays)(i, j);
    ulong ij = 0;
    for (ulong k = 0; k < N_i_size; k++) {
      if (k > 0) {
        for (ulong j1 = 0; j1 < n_nodes; j1++) {
          double beta_j1_j = (*decays)(j1, j);
          H(j1, j) *= cexp(
            -beta_j1_j * ((*timestamps_i)[k] - (*timestamps_i)[k - 1]));
        }
      }
      while ((ij < N_j_size) && ((*realization_j)[ij] < (*timestamps_i)[k])) {
        for (ulong j1 = 0; j1 < n_nodes; j1++) {
          double beta_j1_j = (*decays)(j1, j);
          H(j1, j) += beta_j1_j * cexp(
            -beta_j1_j * ((*timestamps_i)[k] - (*realization_j)[ij]));
        }
        Dg_i[j] += (1 - cexp(-betaij * (end_time - (*realization_j)[ij])));
        Dg2_i[j] += betaij * (1 - cexp(-2 * betaij * (end_time - (*realization_j)[ij]))) / 2;
        ij++;
      }

      C_i[j] += H(i, j);

      // Here we compute E(j1,i,j)
      const ulong index = i * n_nodes + j;
      for (ulong j1 = 0; j1 < n_nodes; j1++) {
        double beta_j1_i = (*decays)(j1, i);
        double beta_j1_j = (*decays)(j1, j);
        ArrayDouble E_j1 = view_row(E, j1);
        double r = beta_j1_i / (beta_j1_i + beta_j1_j);
        E_j1[index] += r * (1 - cexp(-(end_time - (*timestamps_i)[k]) * (beta_j1_i + beta_j1_j)))
          * H(j1, j);
      }
    }

    if (ij < N_j_size) {
      while (ij < N_j_size) {
        Dg2_i[j] += betaij * (1 - cexp(-2 * betaij * (end_time - (*realization_j)[ij]))) / 2;
        Dg_i[j] += (1 - cexp(-betaij * (end_time - (*realization_j)[ij])));
        ij++;
      }
    }
  }
}

ulong ModelHawkesExpKernLeastSqSingle::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
