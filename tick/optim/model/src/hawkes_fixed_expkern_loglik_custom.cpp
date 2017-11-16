// License: BSD 3 clause


#include "hawkes_fixed_expkern_loglik_custom.h"

ModelHawkesCustom::ModelHawkesCustom(
        const double decay, const int max_n_threads) :
        ModelHawkesFixedKernCustom(max_n_threads),
        decay(decay) {}

void ModelHawkesCustom::allocate_weights() {
    if (n_nodes == 0) {
        TICK_ERROR("Please provide valid timestamps before allocating weights")
    }

    //! hacked part of peng Wu
    Total_events = 0;
    for (ulong i = 0; i < n_nodes; i++)
        Total_events += (*n_jumps_per_node)[i];


    g = ArrayDouble2dList1D(n_nodes);
    G = ArrayDouble2dList1D(n_nodes);
    sum_G = ArrayDoubleList1D(n_nodes);

    for (ulong i = 0; i < n_nodes; i++) {
        g[i] = ArrayDouble2d(Total_events + 1, n_nodes);
        g[i].init_to_zero();
        G[i] = ArrayDouble2d(Total_events + 1, n_nodes);
        G[i].init_to_zero();
        sum_G[i] = ArrayDouble(n_nodes);
    }

    global_timestamps = ArrayDouble(Total_events + 1);
    global_timestamps.init_to_zero();
    type_n = ArrayULong(Total_events + 1);
    type_n.init_to_zero();
    global_n = ArrayULong(Total_events + 1);
    global_n.init_to_zero();
}

void ModelHawkesCustom::compute_weights_dim_i(const ulong i, const ArrayDouble &coeffs) {
    const ArrayDouble t_i = view(*timestamps[i]);
    ArrayDouble2d g_i = view(g[i]);
    ArrayDouble2d G_i = view(G[i]);
    ArrayDouble sum_G_i = view(sum_G[i]);


    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));


    //! hacked code here, seperator = 1, meaning L(increasing) is timestamps[1], C,M(decreasing) are timestamps[2] timestamps[3]
    ulong count = 1;
    for (ulong i = 0; i < n_nodes; i++) {
        const ArrayDouble t_i = view(*timestamps[i]);
        for (int j = 0; j < i; j++) {
            global_timestamps[count] = t_i[j];
            type_n[count++] = i + 1;
        }
    }
    global_timestamps.sort(type_n);

    for (int i = 1; i < Total_events + 1; ++i) {
        if (type_n[i] < 2)
            global_n[i] = global_n[i - 1] + 1;
        else
            global_n[i] = global_n[i - 1] - 1;
    }

    //for the g_j, I make all the threads calculating the same g_j in this developing stage
    for (ulong j = 0; j < n_nodes; j++) {
        ulong ij = 0;
        //! here k starts from 1, because g(t_0) = G(t_0) = 0
        for (ulong k = 1; k < Total_events + 1; k++) {
            const double t_k = k < Total_events + 1 ? global_timestamps[k] : end_time;

            const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
            g_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * ebt + type_n[k] == j ? decay : 0;
            G_i[k * n_nodes + j] = (1 - ebt) / decay * g_i[(k - 1) * n_nodes + j] * f_i[global_n[k]];

            sum_G_i[j] += G_i[k * n_nodes + j];
        }
    }
}

ulong ModelHawkesCustom::get_n_coeffs() const {
    //! hacked, how to add a new constant N in the class?
    return n_nodes + n_nodes * n_nodes + n_nodes * MaxN_of_f;
}