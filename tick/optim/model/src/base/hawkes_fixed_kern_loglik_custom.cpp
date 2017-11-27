// License: BSD 3 clause

#include "hawkes_fixed_kern_loglik_custom.h"

ModelHawkesFixedKernCustom::ModelHawkesFixedKernCustom(const ulong _MaxN_of_f, const int max_n_threads) :
        MaxN_of_f(_MaxN_of_f), ModelHawkesSingle(max_n_threads, 0) {}

void ModelHawkesFixedKernCustom::compute_weights() {
    allocate_weights();
    parallel_run(get_n_threads(), n_nodes, &ModelHawkesFixedKernCustom::compute_weights_dim_i, this);
    weights_computed = true;
}

void ModelHawkesFixedKernCustom::allocate_weights() {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

void ModelHawkesFixedKernCustom::compute_weights_dim_i(const ulong i) {
    TICK_CLASS_DOES_NOT_IMPLEMENT("");
}

void ModelHawkesFixedKernCustom::set_data(const SArrayDoublePtrList2D &timestamps_list,
                                          VArrayDoublePtr end_times) {
    if (timestamps_list.size() != 1) TICK_ERROR("Can handle only one realization, provided " << timestamps_list.size());
    ModelHawkesSingle::set_data(timestamps_list[0], (*end_times)[0]);
}

double ModelHawkesFixedKernCustom::loss(const ArrayDouble &coeffs) {
    if (!weights_computed) compute_weights();

    const double loss =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelHawkesFixedKernCustom::loss_dim_i,
                                         this,
                                         coeffs);
    return loss / n_total_jumps;
}

double ModelHawkesFixedKernCustom::loss_i(const ulong sampled_i,
                                          const ArrayDouble &coeffs) {
    if (!weights_computed) compute_weights();
    ulong i;
    ulong k;
    sampled_i_to_index(sampled_i, &i, &k);

    return loss_i_k(i, k, coeffs);
}

void ModelHawkesFixedKernCustom::grad(const ArrayDouble &coeffs,
                                      ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelHawkesFixedKernCustom::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= n_total_jumps;
}

void ModelHawkesFixedKernCustom::grad_i(const ulong sampled_i,
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

double ModelHawkesFixedKernCustom::loss_and_grad(const ArrayDouble &coeffs,
                                                 ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    const double loss =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelHawkesFixedKernCustom::loss_and_grad_dim_i,
                                         this,
                                         coeffs, out);
    out /= n_total_jumps;
    return loss / n_total_jumps;
}

double ModelHawkesFixedKernCustom::hessian_norm(const ArrayDouble &coeffs,
                                                const ArrayDouble &vector) {
    if (!weights_computed) compute_weights();

    const double norm_sum =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelHawkesFixedKernCustom::hessian_norm_dim_i,
                                         this,
                                         coeffs, vector);

    return norm_sum / n_total_jumps;
}

void ModelHawkesFixedKernCustom::hessian(const ArrayDouble &coeffs, ArrayDouble &out) {
    if (!weights_computed) compute_weights();

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(), n_nodes, &ModelHawkesFixedKernCustom::hessian_i,
                 this, coeffs, out);
    out /= n_total_jumps;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////////////////////////

void ModelHawkesFixedKernCustom::sampled_i_to_index(const ulong sampled_i,
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

double ModelHawkesFixedKernCustom::loss_dim_i(const ulong i,
                                              const ArrayDouble &coeffs) {
    const double mu_i = coeffs[i];
    //this->MaxN_of_f = (ulong) coeffs[n_nodes + n_nodes * n_nodes];

    const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));
    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));

    //cozy at hand
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //term 1
    //end_time is T
    double loss = 0;
    for (ulong k = 1; k < Total_events + 1; k++)
        //! insert event t0 = 0 in the Total_events and global_n
        loss += log(f_i[global_n[k]]);

    //term 2
    for (ulong k = 1; k < Total_events + 1; k++) {
        double tmp_s = mu_i;
        const ArrayDouble g_i_k = view_row(g[i], k);
        tmp_s += alpha_i.dot(g_i_k);
        if (tmp_s <= 0) {
            TICK_ERROR("The sum of the influence on someone cannot be negative. "
                               "Maybe did you forget to add a positive constraint to "
                               "your proximal operator");
        }
        loss += log(tmp_s);
    }

    //term 3,4
    for (ulong k = 1; k < Total_events + 1; k++)
        loss -= mu_i * (global_timestamps[k] - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    loss -= mu_i * (end_time - global_timestamps[Total_events]) * f_i[global_n[Total_events]];

    //term 5, 6
    // !sum_g already takes care of the last item T
    for (ulong j = 0; j < n_nodes; j++)
        for (ulong k = 1; k < Total_events + 1; k++) {
            sum_G[i][j] += G_i[k * n_nodes + j] * f_i[global_n[k - 1]];
        }
    loss += alpha_i.dot(sum_G[i]);

    return loss;
}

double ModelHawkesFixedKernCustom::loss_i_k(const ulong i,
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

void ModelHawkesFixedKernCustom::grad_dim_i(const ulong i,
                                            const ArrayDouble &coeffs,
                                            ArrayDouble &out) {
    const double mu_i = coeffs[i];
    double &grad_mu_i = out[i];

    const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));
    ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    ArrayDouble grad_f_i = view(out, get_f_i_first_index(i), get_f_i_last_index(i));

    //necessary information required
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //! grad of mu_i
    for (ulong k = 1; k < Total_events + 1; ++k) {
        double numerator = 1;
        double denominator = mu_i;

        for (ulong j = 0; j < n_nodes; j++) {
            const ArrayDouble g_i_k = view_row(g[i], k);
            denominator += alpha_i.dot(g_i_k);
        }
        grad_mu_i += numerator / denominator;
    }

    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        const double t_k = k < Total_events + 1 ? global_timestamps[k] : end_time;
        grad_mu_i -= (t_k - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    }

    //! grad of alpha_{ij}
    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        const ArrayDouble g_i_k = view_row(g[i], k);
        double s = mu_i + alpha_i.dot(g_i_k);

        grad_alpha_i.mult_incr(g_i_k, 1. / s);
    }
    for (ulong j = 0; j < n_nodes; j++) {
        double tmp = 0;
        for (ulong k = 1; k < 1 + Total_events + 1; k++) {
            tmp += G_i[k] * f_i[global_n[k - 1]];
        }
        grad_alpha_i[j] -= tmp;
    }

    //! grad of f^i_n
    //! in fact, H1_i for different i keep the same information, same thing for H2, H3
    const ArrayDouble H1_i = view(H1[i]);
    const ArrayDouble H2_i = view(H2[i]);
    for (ulong n = 0; n != MaxN_of_f; ++n) {
        const ArrayDouble H3_j_n = view_row(H3, n);
        grad_f_i[n] = H1_i[n] / f_i[n] + mu_i * H2_i[n] + alpha_i.dot(H3_j_n);
    }
}

void ModelHawkesFixedKernCustom::grad_i_k(const ulong i, const ulong k,
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

double ModelHawkesFixedKernCustom::loss_and_grad_dim_i(const ulong i,
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

double ModelHawkesFixedKernCustom::hessian_norm_dim_i(const ulong i,
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

void ModelHawkesFixedKernCustom::hessian_i(const ulong i,
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