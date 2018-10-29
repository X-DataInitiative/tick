// License: BSD 3 clause


#include "hawkes_fixed_sumexpkern_leastsq_qrh1.h"

ModelHawkesFixedSumExpKernLeastSqQRH1::ModelHawkesFixedSumExpKernLeastSqQRH1(
        const ArrayDouble &decays,
        const ulong MaxN,
        const unsigned int max_n_threads,
        const unsigned int optimization_level)
        : ModelHawkesSingle(max_n_threads, optimization_level),
          decays(decays), n_decays(decays.size()), MaxN(MaxN) {}

// Method that computes the value
double ModelHawkesFixedSumExpKernLeastSqQRH1::loss(const ArrayDouble &coeffs) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of the contribution of each component
  SArrayDoublePtr values =
          parallel_map(get_n_threads(),
                       n_nodes,
                       &ModelHawkesFixedSumExpKernLeastSqQRH1::loss_i,
                       this,
                       coeffs);

  // We just need to sum up the contribution
  return values->sum() / n_total_jumps;
}

// Performs the computation of the contribution of the i component to the value
double ModelHawkesFixedSumExpKernLeastSqQRH1::loss_i(const ulong i,
                                                     const ArrayDouble &coeffs) {
    if (!weights_computed) TICK_ERROR("Please compute weights before calling loss_i");

    const double mu_i = coeffs[i];
    ulong U = n_decays;
    ArrayDouble f_i(MaxN);
    for(ulong k = 0; k != MaxN; ++k)
        f_i[k] = coeffs[n_nodes + n_nodes * n_nodes * U + i * MaxN + k];
    double R_i = 0;

    const ArrayDouble2d g_i = view(g[i]);
//    const ArrayDouble2d G_i = view(G[i]);

    //! Term 1
    for(ulong q; q < MaxN; ++q)
        R_i += f_i[q] * f_i[q] * Length[q] * mu_i * mu_i;

    //! Term 2
    auto get_G_index = [=](ulong q, ulong u) {
        return n_decays * q + u;
    };

    double tmp_s = 0;
    for (ulong j = 0; j != n_nodes; ++j) {
        const ArrayDouble2d G_j = view(G[j]);
        for (ulong u = 0; u != U; ++u) {
            double G_ij_u = 0; //! at T
            for (ulong q; q < MaxN; ++q)
                G_ij_u += f_i[q] * f_i[q] * G_j[get_G_index(q, u)];
            double alpha_u_ij = coeffs[get_alpha_u_i_j_index(u, i, j)];
            tmp_s += alpha_u_ij * G_ij_u;
        }
    }
    R_i += 2 * mu_i * tmp_s;

    //! Term 4
    const ArrayULong Count_i = view(Count[i]);
    for(ulong q; q < MaxN; ++q)
        R_i -= 2 * mu_i * f_i[q] * Count_i[q];

    //! Term 5
    auto get_g_index = [=](ulong k, ulong u) {
        return n_decays * k + u;
    };

    for (ulong k = 1; k != n_total_jumps + 1; ++k)
        if (type_n[k] == i + 1) {
            double tmp_s = 0;
            for (ulong j = 0; j != n_nodes; ++j) {
                ArrayDouble2d g_j = view(g[j]);
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    tmp_s += alpha_u_i_j * g_j[get_g_index(k, u)];
                }
            }
            R_i -= 2 * tmp_s * f_i[global_n[k-1]];
        }

    //! Big Term
    auto get_H_index = [=](ulong j, ulong jj, ulong u, ulong uu, ulong q) {
        return  n_nodes * n_decays * n_decays * q + n_decays * n_decays * jj + n_decays * u + uu;
    };

    for (ulong j = 0; j != n_nodes; ++j) {
        ArrayDouble2d H_j = view(H[j]);
        for (ulong u = 0; u != U; ++u) {
            double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
            for (ulong jj = 0; jj != n_nodes; ++jj)
                for (ulong uu = 0; uu != U; ++uu) {
                    double alpha_uu_i_jj = coeffs[get_alpha_u_i_j_index(uu, i, jj)];
                    double tmp_s = 0;
                    for (ulong q; q < MaxN; ++q) {
                        tmp_s += H_j[get_H_index(j, jj, u, uu, q)] * f_i[q] * f_i[q];
                    }
                    R_i += alpha_u_i_j * alpha_uu_i_jj * tmp_s;
                }
        }
    }

    return R_i;
}

// Method that computes the gradient
void ModelHawkesFixedSumExpKernLeastSqQRH1::grad(const ArrayDouble &coeffs,
                                                 ArrayDouble &out) {
  // The initialization should be performed if not performed yet
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesFixedSumExpKernLeastSqQRH1::grad_i,
               this,
               coeffs,
               out);
  out /= n_total_jumps;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}

// Method that computes the component i of the gradient
void ModelHawkesFixedSumExpKernLeastSqQRH1::grad_i(const ulong i,
                                                   const ArrayDouble &coeffs,
                                                   ArrayDouble &out) {
    if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

    //! necessary variables
    const double mu_i = coeffs[i];
    ulong U = n_decays;

    ArrayDouble f_i(MaxN);
    for (ulong q = 0; q != MaxN; ++q)
        f_i[q] = coeffs[n_nodes + n_nodes * n_nodes * U + i * MaxN + q];

    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    auto get_G_index = [=](ulong q, ulong u) {
        return n_decays * q + u;
    };

    auto get_g_index = [=](ulong k, ulong u) {
        return n_decays * k + u;
    };

    auto get_H_index = [=](ulong j, ulong jj, ulong u, ulong uu, ulong q) {
        return n_nodes * n_decays * n_decays * q + n_decays * n_decays * jj + n_decays * u + uu;
    };

    //! grad of mu_i
    double &grad_mu_i = out[i];
    grad_mu_i = 0;

    //! Term 1
    for (ulong q; q < MaxN; ++q)
        grad_mu_i += 2 * mu_i * f_i[q] * f_i[q] * Length[q];

    //! Term 2
    double tmp_s = 0;
    for (ulong j = 0; j != n_nodes; ++j) {
        const ArrayDouble2d G_j = view(G[j]);
        for (ulong u = 0; u != U; ++u) {
            double G_ij_u = 0; //! at T
            for (ulong q; q < MaxN; ++q)
                G_ij_u += f_i[q] * f_i[q] * G_j[get_G_index(q, u)];
            double alpha_u_ij = coeffs[get_alpha_u_i_j_index(u, i, j)];
            tmp_s += alpha_u_ij * G_ij_u;
        }
    }
    grad_mu_i += 2 * tmp_s;

    //! Term 3
    const ArrayULong Count_i = view(Count[i]);
    for (ulong q; q < MaxN; ++q)
        grad_mu_i -= 2 * f_i[q] * Count_i[q];

    //! grad of alpha_u_{ij}, for all j and all u
    //! Term 1
    for (ulong j = 0; j != n_nodes; ++j) {
        const ArrayDouble2d G_j = view(G[j]);
        for (ulong u = 0; u != U; ++u) {
            double G_ij_u = 0;
            for (ulong q; q < MaxN; ++q)
                G_ij_u += f_i[q] * f_i[q] * G_j[get_G_index(q, u)];

            double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
            grad_alpha_u_ij = 2 * mu_i * G_ij_u;
        }
    }

    //! Term 2
    for (ulong k = 1; k != n_total_jumps + 1; ++k)
        if (type_n[k] == i + 1)
            for (ulong j = 0; j != n_nodes; ++j) {
                ArrayDouble2d g_j = view(g[j]);
                for (ulong u = 0; u != U; ++u) {
                    double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
                    grad_alpha_u_ij -= f_i[global_n[k - 1]] * g_j[get_g_index(k, u)];;
                }
            }

    //! Term 3
    for (ulong j = 0; j != n_nodes; ++j) {
        ArrayDouble2d H_j = view(H[j]);
        for (ulong u = 0; u != U; ++u) {
            double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
            for (ulong jj = 0; jj != n_nodes; ++jj)
                for (ulong uu = 0; uu != U; ++uu) {
                    double tmp_s = 0;
                    for (ulong q; q < MaxN; ++q)
                        tmp_s += H_j[get_H_index(j, jj, u, uu, q)] * f_i[q] * f_i[q];

                    double alpha_uu_i_jj = coeffs[get_alpha_u_i_j_index(uu, i, jj)];
                    grad_alpha_u_ij += alpha_uu_i_jj * tmp_s;
                }
        }
    }

    //! grad of f^i_n
    //! Term 1
    ArrayDouble grad_f_i(MaxN);
    for (ulong q; q < MaxN; ++q)
        grad_f_i[q] = 2 * f_i[q] * Length[q] * mu_i * mu_i;

    //! Term 2
    //! to be finished
    for (ulong j = 0; j != n_nodes; ++j) {
        const ArrayDouble2d G_j = view(G[j]);
        for (ulong u = 0; u != U; ++u) {
            double alpha_u_ij = coeffs[get_alpha_u_i_j_index(u, i, j)];
            for (ulong q; q < MaxN; ++q)
                grad_f_i[q] += 4 * f_i[q] * alpha_u_ij * G_j[get_G_index(q, u)];
        }
    }

    //! Term 3
    for(ulong q; q < MaxN; ++q)
        grad_f_i[q] -= 2 * mu_i * Count_i[q];

    //! Term 4
    for (ulong k = 1; k != n_total_jumps + 1; ++k)
        if (type_n[k] == i + 1) {
            double tmp_s = 0;
            for (ulong j = 0; j != n_nodes; ++j) {
                ArrayDouble2d g_j = view(g[j]);
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    tmp_s += alpha_u_i_j * g_j[get_g_index(k, u)];
                }
            }
            grad_f_i[global_n[k-1]] -= 2 * tmp_s;
        }

    //! Big Term
    for (ulong j = 0; j != n_nodes; ++j) {
        ArrayDouble2d H_j = view(H[j]);
        for (ulong u = 0; u != U; ++u) {
            double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
            for (ulong jj = 0; jj != n_nodes; ++jj)
                for (ulong uu = 0; uu != U; ++uu) {
                    double alpha_uu_i_jj = coeffs[get_alpha_u_i_j_index(uu, i, jj)];
                    for (ulong q; q < MaxN; ++q)
                        grad_f_i[q] += alpha_u_i_j * alpha_uu_i_jj * H_j[get_H_index(j, jj, u, uu, q)] * 2 * f_i[q];
                }
        }
    }

    //Write to the out vector
    for(ulong q = 0; q != MaxN; ++q)
        out[n_nodes + n_nodes * n_nodes * U + i * MaxN + q] = grad_f_i[q];
}

// Computes both gradient and value
double ModelHawkesFixedSumExpKernLeastSqQRH1::loss_and_grad(const ArrayDouble &coeffs,
                                                            ArrayDouble &out) {
  grad(coeffs, out);
  return loss(coeffs);
}

// Contribution of the ith component to the initialization
// Computation of the arrays H, Dg, Dg2 and C
void ModelHawkesFixedSumExpKernLeastSqQRH1::compute_weights_i(const ulong i) {
    //!thread i computes weights governed by dimension i

    //! Length(n) and Count^i(n)
    for (ulong k = 1; k != 1 + n_total_jumps; ++k) {
        const double delta_t = global_timestamps[k] - global_timestamps[k - 1];
        const ulong q = global_n[k - 1];

        if (i == 0) //!thread 0
            Length[q] += delta_t;
        if (i == type_n[k] - 1) //!thread i
            Count[i][q]++;
    }
    ArrayDouble2d g_i = view(g[i]);
    ArrayDouble2d G_i = view(G[i]);

    auto get_g_index = [=](ulong k, ulong u) {
        return n_decays * k + u;
    };

    auto get_G_index = [=](ulong q, ulong u) {
        return n_decays * q + u;
    };

    //! computation of g^j_u
    //! computation of G^j_u(n)
    for (ulong u = 0; u != n_decays; ++u) {
        double decay = decays[u];
        //! here k starts from 1, cause g(t_0) = G(t_0) = 0
        //! 0 + n_total_jumps + T
        for (ulong k = 1; k != 1 + n_total_jumps + 1; k++) {
            const double t_k = (k != (1 + n_total_jumps) ? global_timestamps[k] : end_time);
            const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
            if (k != 1 + n_total_jumps)
                g_i[get_g_index(k, u)] = g_i[get_g_index(k - 1, u)] * ebt + (type_n[k - 1] == i + 1 ? decay * ebt : 0);

            if (k != (1 + n_total_jumps))//!T
                if (i + 1 == type_n[k]) //!! so we need more n_threads than dim?
                    G_i[get_G_index(global_n[k - 1], u)] +=
                            (1 - ebt) / decay * g_i[get_g_index(k - 1, u)] + ((type_n[k - 1] == i + 1) ? 1 - ebt : 0);
        }
    }
}

void ModelHawkesFixedSumExpKernLeastSqQRH1::compute_weights_H_j(const ulong j){
    auto get_g_index = [=](ulong k, ulong u) {
        return n_decays * k + u;
    };

    auto get_H_index = [=](ulong j, ulong jj, ulong u, ulong uu, ulong q) {
        return  n_nodes * n_decays * n_decays * q + n_decays * n_decays * jj + n_decays * u + uu;
    };

    ulong U = decays.size();
    ArrayDouble2d g_j = view(g[j]);
    ArrayDouble2d H_j = view(H[j]);

    //! computation of H^jj'_uu'(n)
    for(ulong jj = 0; jj != n_nodes; jj++) {
        ArrayDouble2d g_jj = view(g[jj]);
        for (ulong u = 0; u != U; ++u)
            for (ulong uu = 0; uu != U; ++uu)
                for (ulong k = 1; k != 1 + n_total_jumps + 1; k++) {
                    const double decay = decays[u] + decays[uu];
                    const double t_k = (k != (1 + n_total_jumps) ? global_timestamps[k] : end_time);
                    const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
                    const double x0 = g_j[get_g_index(k, u)] * g_jj[get_g_index(k, uu)] / ebt;

                    const ulong q = global_n[k - 1];
                    H_j[get_H_index(j, jj, u, uu, q)] += (1 - ebt) / decay * x0;
                }
    }
}

void ModelHawkesFixedSumExpKernLeastSqQRH1::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }

  Total_events = n_total_jumps - (*n_jumps_per_node)[n_nodes];

  //! g^j_u for all t_k
  g = ArrayDouble2dList1D(n_nodes);
  //! G^j_u for all state in [0, MaxN[
  G = ArrayDouble2dList1D(n_nodes);
  H = ArrayDouble2dList1D(n_nodes);

  Length = ArrayDouble(MaxN);
  Length.init_to_zero();
  Count = ArrayULongList1D(n_nodes);

  for (ulong i = 0; i != n_nodes; i++) {
    //0 + events + T
    g[i] = ArrayDouble2d(n_total_jumps + 2, n_decays);
    g[i].init_to_zero();
    G[i] = ArrayDouble2d(MaxN, n_decays);
    G[i].init_to_zero();
    Count[i] = ArrayULong(MaxN);
    Count[i].init_to_zero();

    H[i] = ArrayDouble2d(n_nodes * n_decays * n_decays, MaxN);
    H[i].init_to_zero();
  }
}

// Weights should be computed before loss and grad
void ModelHawkesFixedSumExpKernLeastSqQRH1::compute_weights() {
  allocate_weights();

  // Multithreaded computation of the arrays
  parallel_run(get_n_threads(), n_nodes,
               &ModelHawkesFixedSumExpKernLeastSqQRH1::compute_weights_i,
               this);

    //! H could only be computed after we have all g_i
  parallel_run(get_n_threads(), n_nodes,
               &ModelHawkesFixedSumExpKernLeastSqQRH1::compute_weights_H_j,
               this);
  weights_computed = true;
}

ulong ModelHawkesFixedSumExpKernLeastSqQRH1::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * n_decays + n_nodes * MaxN;
}


void ModelHawkesFixedSumExpKernLeastSqQRH1::set_data(const SArrayDoublePtrList1D &_timestamps,
                                       const SArrayLongPtr _global_n,
                                       const double _end_times){
  ModelHawkesSingle::set_data(_timestamps, _end_times);

  //! create state according to sorting of timestamps
  global_n = ArrayLong(n_total_jumps + 1);
  for(ulong k = 0; k != n_total_jumps + 1; ++k)
    global_n[k] = _global_n->value(k);

  ArrayULong tmp_pre_type_n(n_total_jumps + 1);
  tmp_pre_type_n[0] = 0;
  ArrayULong tmp_index(n_total_jumps + 1);

  global_timestamps = ArrayDouble(n_total_jumps + 1);
  global_timestamps.init_to_zero();
  type_n = ArrayULong(n_total_jumps + 1);
  type_n.init_to_zero();

  ulong count = 1;
  for (ulong j = 0; j != n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps[j]);
    for (ulong k = 0; k != (*n_jumps_per_node)[j]; ++k) {
      global_timestamps[count] = t_j[k];
      tmp_pre_type_n[count++] = j + 1;
    }
  }

  global_timestamps.sort(tmp_index);

  for (ulong k = 1; k != n_total_jumps + 1; ++k)
    type_n[k] = tmp_pre_type_n[tmp_index[k]];

  n_nodes--;
}