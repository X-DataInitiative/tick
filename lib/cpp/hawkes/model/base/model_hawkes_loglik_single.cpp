// License: BSD 3 clause

#include "tick/hawkes/model/base/model_hawkes_loglik_single.h"

ModelHawkesLogLikSingle::ModelHawkesLogLikSingle(const int max_n_threads) :
  ModelHawkesSingle(max_n_threads, 0) {}

void ModelHawkesLogLikSingle::compute_weights() {
  allocate_weights();
  parallel_run(get_n_threads(), n_nodes, &ModelHawkesLogLikSingle::compute_weights_dim_i, this);
  weights_computed = true;
}

void ModelHawkesLogLikSingle::allocate_weights() {
  TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

void ModelHawkesLogLikSingle::compute_weights_dim_i(const ulong i) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

double ModelHawkesLogLikSingle::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();

  const double loss =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesLogLikSingle::loss_dim_i,
                                 this,
                                 coeffs);
  return loss / n_total_jumps;
}

double ModelHawkesLogLikSingle::loss_i(const ulong sampled_i,
                                       const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  ulong i;
  ulong k;
  sampled_i_to_index(sampled_i, &i, &k);

  return loss_i_k(i, k, coeffs);
}

void ModelHawkesLogLikSingle::grad(const ArrayDouble &coeffs,
                                   ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  out.fill(0);

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesLogLikSingle::grad_dim_i,
               this,
               coeffs,
               out);
  out /= n_total_jumps;
}

void ModelHawkesLogLikSingle::grad_i(const ulong sampled_i,
                                     const ArrayDouble &coeffs,
                                     ArrayDouble &out) {
  if (!weights_computed) compute_weights();

  ulong i;
  ulong k;
  sampled_i_to_index(sampled_i, &i, &k);

  // set grad to zero
  out.fill(0);

  grad_i_k(i, k, coeffs, out);
}

double ModelHawkesLogLikSingle::loss_and_grad(const ArrayDouble &coeffs,
                                              ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  out.fill(0);

  const double loss =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesLogLikSingle::loss_and_grad_dim_i,
                                 this,
                                 coeffs, out);
  out /= n_total_jumps;
  return loss / n_total_jumps;
}

double ModelHawkesLogLikSingle::hessian_norm(const ArrayDouble &coeffs,
                                             const ArrayDouble &vector) {
  if (!weights_computed) compute_weights();

  const double norm_sum =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesLogLikSingle::hessian_norm_dim_i,
                                 this,
                                 coeffs, vector);

  return norm_sum / n_total_jumps;
}

void ModelHawkesLogLikSingle::hessian(const ArrayDouble &coeffs, ArrayDouble &out) {
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(), n_nodes, &ModelHawkesLogLikSingle::hessian_i,
               this, coeffs, out);
  out /= n_total_jumps;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////////////////////////

void ModelHawkesLogLikSingle::sampled_i_to_index(const ulong sampled_i,
                                                 ulong *i,
                                                 ulong *k) {
  ulong cum_N_i = 0;
  for (ulong d = 0; d < n_nodes; d++) {
    cum_N_i += (*n_jumps_per_node)[d];
    if (sampled_i < cum_N_i) {
      *i = d;
      *k = sampled_i - cum_N_i + (*n_jumps_per_node)[d];
      break;
    }
  }
}

double ModelHawkesLogLikSingle::loss_dim_i(const ulong i,
                                           const ArrayDouble &coeffs) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double loss = -end_time;
  loss += end_time * mu_i;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);
    if (s <= 0) {
      TICK_ERROR("The sum of the influence on someone cannot be negative. "
                   "Maybe did you forget to add a positive constraint to "
                   "your proximal operator");
    }
    loss -= log(s);
  }

  loss += alpha_i.dot(sum_G[i]);
  return loss;
}

double ModelHawkesLogLikSingle::loss_i_k(const ulong i,
                                         const ulong k,
                                         const ArrayDouble &coeffs) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));
  double loss = 0;

  const ArrayDouble g_i_k = view_row(g[i], k);
  const ArrayDouble G_i_k = view_row(G[i], k);

  // Both are correct, just a question of point of view
  const double t_i_k = k == (*n_jumps_per_node)[i] - 1 ? end_time : (*timestamps[i])[k];
  const double t_i_k_minus_one = k == 0 ? 0 : (*timestamps[i])[k - 1];
  loss += (t_i_k - t_i_k_minus_one) * (mu_i - 1);
  //  loss += end_time * (mu[i] - 1) / (*n_jumps_per_node)[i];

  double s = mu_i;
  s += alpha_i.dot(g_i_k);

  if (s <= 0) {
    TICK_ERROR("The sum of the influence on someone cannot be negative. Maybe did "
                 "you forget to add a positive constraint to your "
                 "proximal operator");
  }
  loss -= log(s);

  loss += alpha_i.dot(G_i_k);
  if (k == (*n_jumps_per_node)[i] - 1)
    loss += alpha_i.dot(view_row(G[i], k + 1));

  return loss;
}

void ModelHawkesLogLikSingle::grad_dim_i(const ulong i,
                                         const ArrayDouble &coeffs,
                                         ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  grad_mu_i += end_time;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);
    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    grad_mu_i -= 1. / s;
    grad_alpha_i.mult_incr(g_i_k, -1. / s);
  }

  grad_alpha_i.mult_incr(sum_G[i], 1);
}

void ModelHawkesLogLikSingle::grad_i_k(const ulong i, const ulong k,
                                       const ArrayDouble &coeffs,
                                       ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  const ArrayDouble g_i_k = view_row(g[i], k);
  const ArrayDouble G_i_k = view_row(G[i], k);

  // Both are correct, just a question of point of view
  const double t_i_k = k == (*n_jumps_per_node)[i] - 1 ? end_time : (*timestamps[i])[k];
  const double t_i_k_minus_one = k == 0 ? 0 : (*timestamps[i])[k - 1];
  grad_mu_i += t_i_k - t_i_k_minus_one;
  //  grad_mu[i] += end_time / (*n_jumps_per_node)[i];

  double s = mu_i;
  s += alpha_i.dot(g_i_k);

  grad_mu_i -= 1. / s;
  grad_alpha_i.mult_incr(g_i_k, -1. / s);
  grad_alpha_i.mult_incr(G_i_k, 1.);

  if (k == (*n_jumps_per_node)[i] - 1)
    grad_alpha_i.mult_incr(view_row(G[i], k + 1), 1.);
}

double ModelHawkesLogLikSingle::loss_and_grad_dim_i(const ulong i,
                                                    const ArrayDouble &coeffs,
                                                    ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double loss = 0;

  grad_mu_i += end_time;
  loss += end_time * mu_i;
  for (ulong k = 0; k < (*n_jumps_per_node)[i]; k++) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    if (s <= 0) {
      TICK_ERROR("The sum of the influence on someone cannot be negative. Maybe did "
                   "you forget to add a positive constraint to your "
                   "proximal operator");
    }
    loss -= log(s);
    grad_mu_i -= 1. / s;

    grad_alpha_i.mult_incr(g_i_k, -1. / s);
  }

  loss += alpha_i.dot(sum_G[i]);
  grad_alpha_i.mult_incr(sum_G[i], 1);

  return loss;
}

double ModelHawkesLogLikSingle::hessian_norm_dim_i(const ulong i,
                                                   const ArrayDouble &coeffs,
                                                   const ArrayDouble &vector) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double d_mu_i = vector[i];
  ArrayDouble d_alpha_i = view(vector, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double hess_norm = 0;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; k++) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double S = d_mu_i;
    S += d_alpha_i.dot(g_i_k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    double tmp = S / s;
    hess_norm += tmp * tmp;
  }
  return hess_norm;
}

void ModelHawkesLogLikSingle::hessian_i(const ulong i,
                                        const ArrayDouble &coeffs,
                                        ArrayDouble &out) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  // number of alphas per dimension
  const ulong n_alpha_i = get_alpha_i_last_index(i) - get_alpha_i_first_index(i);

  const ulong start_mu_line = i * (n_alpha_i + 1);
  const ulong block_start = (n_nodes + i * n_alpha_i) * (n_alpha_i + 1);

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);
    const double s_2 = s * s;

    // fill mu mu
    out[start_mu_line] += 1. / s_2;
    // fill mu alpha
    for (ulong j = 0; j < n_alpha_i; ++j) {
      out[start_mu_line + j + 1] += g_i_k[j] / s_2;
    }

    for (ulong l = 0; l < n_alpha_i; ++l) {
      const ulong start_alpha_line = block_start + l * (n_alpha_i + 1);
      // fill alpha mu
      out[start_alpha_line] += g_i_k[l] / s_2;
      // fill alpha square
      for (ulong m = 0; m < n_alpha_i; ++m) {
        out[start_alpha_line + m + 1] += g_i_k[l] * g_i_k[m] / s_2;
      }
    }
  }
}
