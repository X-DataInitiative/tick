// License: BSD 3 clause


#include "hawkes_fixed_expkern_loglik_custom2.h"

ModelHawkesCustomType2::ModelHawkesCustomType2(const double _decay, const ulong _MaxN, const int max_n_threads) :
        MaxN(_MaxN), ModelHawkesFixedKernLogLik(max_n_threads), decay(_decay) {}

void ModelHawkesCustomType2::allocate_weights() {
    if (n_nodes == 0) {
        TICK_ERROR("Please provide valid timestamps before allocating weights")
    }

    Total_events = n_total_jumps;

    g = ArrayDouble2dList1D(n_nodes);
    G = ArrayDouble2dList1D(n_nodes);
    sum_G = ArrayDoubleList1D(n_nodes);

    for (ulong i = 0; i != n_nodes; i++) {
        //0 + events + T
        g[i] = ArrayDouble2d(Total_events + 2, n_nodes);
        g[i].init_to_zero();
        G[i] = ArrayDouble2d(Total_events + 2, n_nodes);
        G[i].init_to_zero();
        sum_G[i] = ArrayDouble(n_nodes);
        sum_G[i].init_to_zero();
    }
}

void ModelHawkesCustomType2::compute_weights_dim_i(const ulong i) {
    ArrayDouble2d g_i = view(g[i]);
    ArrayDouble2d G_i = view(G[i]);

    //for the g_j, I make all the threads calculating the same g in this developing stage
    for (ulong j = 0; j != n_nodes; j++) {
        //! here k starts from 1, cause g(t_0) = G(t_0) = 0
        // 0 + Totalevents + T
        for (ulong k = 1; k != 1 + Total_events + 1; k++) {
            const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
            const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
            if (k != 1 + Total_events)
                g_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * ebt + (type_n[k - 1] == j + 1 ? decay * ebt : 0);
            G_i[k * n_nodes + j] =
                    (1 - ebt) / decay * g_i[(k - 1) * n_nodes + j] + ((type_n[k - 1] == j + 1) ? 1 - ebt : 0);

            // ! in the G, we calculated the difference between G without multiplying f
            // sum_G is calculated later, in the calculation of L_dim_i and its grads
        }
    }
}

void ModelHawkesCustomType2::set_data(const SArrayDoublePtrList1D &_timestamps,
                                 const SArrayLongPtr _global_n,
                                 const double _end_times){
    ModelHawkesSingle::set_data(_timestamps, _end_times);

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

double ModelHawkesCustomType2::loss_dim_i(const ulong i,
                                              const ArrayDouble &coeffs) {
    double loss = 0;

    const ArrayDouble mu_i = view(coeffs, get_mu_i_first_index(i), get_mu_i_last_index(i));
    const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

    //cozy at hand
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //term 1
    for (ulong k = 1; k != Total_events + 1; k++)
        if (type_n[k] == i + 1) {
            double tmp_s = mu_i[global_n[k - 1]];
            const ArrayDouble g_i_k = view_row(g[i], k);
            tmp_s += alpha_i.dot(g_i_k);
            if (tmp_s <= 0) {
                TICK_ERROR("The sum of the influence on someone cannot be negative. "
                                   "Maybe did you forget to add a positive constraint to "
                                   "your proximal operator, in ModelHawkesCustomType2::loss_dim_i");
            }
            loss += log(tmp_s);
        }

    //term 2, 3
    for (ulong k = 1; k != Total_events + 1; k++)
        loss -= mu_i[global_n[k - 1]] * (global_timestamps[k] - global_timestamps[k - 1]);
    loss -= mu_i[global_n[Total_events]] * (end_time - global_timestamps[Total_events]);

    //! clean sum_G each time
    sum_G[i].init_to_zero();

    //term 4,5
    //! sum_g already takes care of the last item T
    for (ulong j = 0; j != n_nodes; j++)
        for (ulong k = 1; k != 1 + Total_events + 1; k++) {
            sum_G[i][j] += G_i[k * n_nodes + j];
        }
    loss -= alpha_i.dot(sum_G[i]);

    //add a constant to the loss, then inverse the loss to make it convex
    return -end_time - loss;
}

void ModelHawkesCustomType2::grad(const ArrayDouble &coeffs,
                                      ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelHawkesCustomType2::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= n_total_jumps;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}

void ModelHawkesCustomType2::grad_dim_i(const ulong i,
                                            const ArrayDouble &coeffs,
                                            ArrayDouble &out) {
    const ArrayDouble mu_i = view(coeffs, get_mu_i_first_index(i), get_mu_i_last_index(i));
    ArrayDouble grad_mu_i = view(out, get_mu_i_first_index(i), get_mu_i_last_index(i));


    const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));
    ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

    //necessary information
    const ArrayDouble2d g_i = view(g[i]);
    const ArrayDouble2d G_i = view(G[i]);

    //! grad of mu_i(n)
    for (ulong k = 1; k < Total_events + 1; ++k) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            //! recall that all g_i are the same
            const ArrayDouble g_i_k = view_row(g[i], k);
            double denominator = mu_i[global_n[k - 1]] + alpha_i.dot(g_i_k);
            grad_mu_i[global_n[k - 1]] += 1.0 / denominator;
        }
    }

    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        const double t_k = (k != (Total_events + 1)) ? global_timestamps[k] : end_time;
        grad_mu_i[global_n[k - 1]] -= (t_k - global_timestamps[k - 1]);
    }

    //! grad of alpha_{ij}
    for (ulong k = 1; k < 1 + Total_events + 1; k++) {
        int tmp_flag = 0;
        if (k == Total_events + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            const ArrayDouble g_i_k = view_row(g[i], k);
            double s = mu_i[global_n[k - 1]] + alpha_i.dot(g_i_k);

            grad_alpha_i.mult_incr(g_i_k, 1. / s);
        }
    }
    for (ulong j = 0; j < n_nodes; j++) {
        double sum_G_ij = 0;
        for (ulong k = 1; k < 1 + Total_events + 1; k++) {
            sum_G_ij += G_i[k * n_nodes + j];
        }
        grad_alpha_i[j] -= sum_G_ij;
    }
}
