// License: BSD 3 clause

#include "hawkes_fixed_sumexpkern_loglik_custom.h"

ModelHawkesFixedSumExpKernCustom::ModelHawkesFixedSumExpKernCustom(const ulong _MaxN_of_f, const ArrayDouble _decays, const int max_n_threads):
        MaxN_of_f(_MaxN_of_f), ModelHawkesSingle(max_n_threads, 0), decays(_decays) {}

void ModelHawkesFixedSumExpKernCustom::compute_weights() {
    allocate_weights();
    parallel_run(get_n_threads(), n_nodes, &ModelHawkesFixedSumExpKernCustom::compute_weights_dim_i, this);
    weights_computed = true;
}

void ModelHawkesFixedSumExpKernCustom::allocate_weights() {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

void ModelHawkesFixedSumExpKernCustom::compute_weights_dim_i(const ulong i) {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

//void ModelHawkesFixedSumExpKernCustom::set_data(const SArrayDoublePtrList2D &timestamps_list,
//                                          VArrayDoublePtr end_times) {
//    if (timestamps_list.size() != 1) TICK_ERROR("Can handle only one realization, provided " << timestamps_list.size());
//    ModelHawkesSingle::set_data(timestamps_list[0], (*end_times)[0]);
//}

double ModelHawkesFixedSumExpKernCustom::loss(const ArrayDouble &coeffs) {
    if (!weights_computed) compute_weights();

    const double loss =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelHawkesFixedSumExpKernCustom::loss_dim_i,
                                         this,
                                         coeffs);
    return loss / n_total_jumps;
}

void ModelHawkesFixedSumExpKernCustom::grad(const ArrayDouble &coeffs,
                                      ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelHawkesFixedSumExpKernCustom::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= n_total_jumps;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}


////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////////////////////////

double ModelHawkesFixedSumExpKernCustom::loss_dim_i(const ulong i,
                                              const ArrayDouble &coeffs) {
    const double mu_i = coeffs[i];
    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    double loss = 0;

    ulong U = this->decays.size();
    //cozy at hand
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    auto get_index = [=](ulong k, ulong j, ulong u) {
        return n_nodes * decays.size() * k + decays.size() * j + u;
    };

    //term 1
    //end_time is T
    for (ulong k = 1; k != Total_events + 1; ++k)
        //! insert event t0 = 0 in the Total_events and global_n
        if (type_n[k] == i + 1)
            loss += log(f_i[global_n[k - 1]]);

    //term 2
    for (ulong k = 1; k != Total_events + 1; ++k)
        if (type_n[k] == i + 1) {
            double tmp_s = mu_i;
            for(ulong j = 0; j != n_nodes; ++j)
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    tmp_s += alpha_u_i_j * g_i[get_index(k, j, u)];
                }

            if (tmp_s <= 0) {
                printf("\nDebug Info : %d %d\n", i, k);
                    TICK_ERROR("The sum of the influence on someone cannot be negative. "
                                   "Maybe did you forget to add a positive constraint to "
                                   "your proximal operator, in SumExp::loss_dim_i");
            }
            loss += log(tmp_s);
        }

    //term 3,4
    for (ulong k = 1; k != Total_events + 1; k++)
        loss -= mu_i * (global_timestamps[k] - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    loss -= mu_i * (end_time - global_timestamps[Total_events]) * f_i[global_n[Total_events]];

    //! clean sum_G each time
    sum_G[i].init_to_zero();

    //term 5, 6
    //! sum_g already takes care of the last item T
    for (ulong j = 0; j != n_nodes; j++) {
        for (ulong u = 0; u != U; ++u) {
            double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
            double sum_G_i_j_u = 0;
            for (ulong k = 1; k != 1 + Total_events + 1; k++) {
                sum_G_i_j_u += G_i[get_index(k, j, u)] * f_i[global_n[k - 1]];
            }
            loss -= alpha_u_i_j * sum_G_i_j_u;
        }
    }

    return -end_time - loss;
}

void ModelHawkesFixedSumExpKernCustom::grad_dim_i(const ulong i,
                                            const ArrayDouble &coeffs,
                                            ArrayDouble &out) {
    const double mu_i = coeffs[i];
    double &grad_mu_i = out[i];

    ulong U = decays.size();
    auto get_index = [=](ulong k, ulong j, ulong u) {
        return n_nodes * decays.size() * k + decays.size() * j + u;
    };

    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    ArrayDouble grad_f_i = view(out, get_f_i_first_index(i), get_f_i_last_index(i));

    //necessary information required
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //! grad of mu_i
    grad_mu_i = 0;
    for (ulong k = 1; k < Total_events + 1; ++k) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            //! recall that all g_i are the same
            double denominator = mu_i;
            for (ulong j = 0; j != n_nodes; j++)
                for (ulong u = 0; u != U; ++u) {
                    double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                    denominator += alpha_u_i_j * g_i[get_index(k, j, u)];
            }
            grad_mu_i += 1 / denominator;
        }
    }

    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        const double t_k = (k != (Total_events + 1)) ? global_timestamps[k] : end_time;
        grad_mu_i -= (t_k - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    }

    //! grad of alpha_u_{ij}
    //! here we calculate the grad of alpha_ij_u, for all j and all u
    for (ulong k = 1; k < 1 + Total_events + 1; ++k) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            //calculate the denominator
            double denominator = mu_i;
            for (ulong jj = 0; jj != n_nodes; ++jj)
                for (ulong uu = 0; uu != U; ++uu) {
                    double alpha_uu_i_jj = coeffs[get_alpha_u_i_j_index(uu, i, jj)];
                    denominator += alpha_uu_i_jj * g_i[get_index(k, jj, uu)];
                }
            for (ulong j = 0; j != n_nodes; ++j)
                for (ulong u = 0; u != U; ++u) {
                    double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
                    grad_alpha_u_ij += g_i[get_index(k, j, u)] / denominator;
                }
        }
    }

    for (ulong j = 0; j != n_nodes; ++j)
        for (ulong u = 0; u != U; ++u) {
            double sum_G_i_j_u = 0;
            for (ulong k = 1; k != 1 + Total_events + 1; k++) {
                sum_G_i_j_u += G_i[get_index(k, j, u)] * f_i[global_n[k - 1]];
            }
            double &grad_alpha_u_ij = out[get_alpha_u_i_j_index(u, i, j)];
            grad_alpha_u_ij -= sum_G_i_j_u;
        }

    //! grad of f^i_n
    //! in fact, H1_i for different i keep the same information, same thing for H2, H3
    const ArrayDouble H1_i = view(H1[i]);
    const ArrayDouble H2_i = view(H2[i]);
    for (ulong n = 0; n != MaxN_of_f; ++n) {
        double result_dot = 0; //! alpha_i_j_u.dot(H3_j_u_n);
        for (ulong j = 0; j != n_nodes; ++j) {
            const ArrayDouble H3_j = view(H3[j]);
            for (ulong u = 0; u != U; ++u) {
                double alpha_u_i_j = coeffs[get_alpha_u_i_j_index(u, i, j)];
                result_dot += alpha_u_i_j * H3_j[n * U + u];
            }
        }
        grad_f_i[n] = H1_i[n] / f_i[n] + mu_i * H2_i[n] + result_dot;
    }
}
