// License: BSD 3 clause


#include "hawkes_fixed_expkern_loglik_custom.h"

ModelHawkesCustom::ModelHawkesCustom(const double _decay, const ulong _MaxN_of_f, const int max_n_threads) :
        ModelHawkesFixedKernCustom(_MaxN_of_f, max_n_threads),
        decay(_decay) {}

void ModelHawkesCustom::allocate_weights() {
    if (n_nodes == 0) {
        TICK_ERROR("Please provide valid timestamps before allocating weights")
    }

    //! hacked part of peng Wu
    Total_events = 0;
    for (ulong i = 0; i != n_nodes; i++)
        Total_events += (*n_jumps_per_node)[i];

    g = ArrayDouble2dList1D(n_nodes);
    G = ArrayDouble2dList1D(n_nodes);
    sum_G = ArrayDoubleList1D(n_nodes);

    H1 = ArrayDoubleList1D(n_nodes);
    H2 = ArrayDoubleList1D(n_nodes);
    H3 = ArrayDoubleList1D(n_nodes);

    for (ulong i = 0; i != n_nodes; i++) {
        //0 + events + T
        g[i] = ArrayDouble2d(Total_events + 2, n_nodes);
        g[i].init_to_zero();
        G[i] = ArrayDouble2d(Total_events + 2, n_nodes);
        G[i].init_to_zero();
        sum_G[i] = ArrayDouble(n_nodes);
        sum_G[i].init_to_zero();

        H1[i] = ArrayDouble(MaxN_of_f);
        H1[i].init_to_zero();
        H2[i] = ArrayDouble(MaxN_of_f);
        H2[i].init_to_zero();
        H3[i] = ArrayDouble(MaxN_of_f);
        H3[i].init_to_zero();
    }

    global_timestamps = ArrayDouble(Total_events + 1);
    global_timestamps.init_to_zero();
    type_n = ArrayULong(Total_events + 1);
    type_n.init_to_zero();
    global_n = ArrayULong(Total_events + 1);
    global_n.init_to_zero();
}

void ModelHawkesCustom::compute_weights_dim_i(const ulong i) {
    ArrayDouble2d g_i = view(g[i]);
    ArrayDouble2d G_i = view(G[i]);

    //! hacked code here, seperator = 1, meaning L(increasing) is timestamps[1], C,M(decreasing) are timestamps[2] timestamps[3]
    ArrayULong tmp_pre_type_n(Total_events + 1);
    tmp_pre_type_n[0] = 0;
    ArrayULong tmp_index(Total_events + 1);

    ulong count = 1;
    for (ulong j = 0; j != n_nodes; j++) {
        const ArrayDouble t_j = view(*timestamps[j]);
        for (ulong k = 0; k != (*n_jumps_per_node)[j]; ++k) {
            global_timestamps[count] = t_j[k];
            tmp_pre_type_n[count++] = j + 1;
        }
    }

    global_timestamps.sort(tmp_index);

    for (ulong k = 1; k != Total_events + 1; ++k) {
        type_n[k] = tmp_pre_type_n[tmp_index[k]];
        if (type_n[k] < 2)
            global_n[k] = global_n[k - 1] + 1;
        else
            global_n[k] = global_n[k - 1] - 1;
    }

    //! ######################
    //! Martin's timestamps make global_n negative sometimes, in this test phase, let's make them compulsoryly position by taking abs
    for (ulong k = 1; k != Total_events + 1; ++k)
        global_n[k] = abs(global_n[k]);
    //! end of hacking part
    //! ######################

//    printf("\n\n");
//    for (ulong k = 0; k < Total_events + 1; ++k)
//        printf("%d %d %f\n", type_n[k],global_n[k],global_timestamps[k]);
//    printf("\n\n");

    //for the g_j, I make all the threads calculating the same g in this developing stage
    for (ulong j = 0; j != n_nodes; j++) {
        //! here k starts from 1, cause g(t_0) = G(t_0) = 0
        // 0 + Totalevents + T
        for (ulong k = 1; k != 1 + Total_events + 1; k++) {
            const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
            const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
            if (k != 1 + Total_events)
                g_i[k * n_nodes + j] = g_i[(k - 1) * n_nodes + j] * ebt + (type_n[k] == j + 1 ? decay : 0);
            G_i[k * n_nodes + j] = (1 - ebt) / decay * g_i[(k - 1) * n_nodes + j];

            // ! in the G, we calculated the difference between G without multiplying f
            // sum_G is calculated later, in the calculation of L_dim_i and its grads
            // debug printf("%d %d %d %d %f #############%f %f\n",i,j,k,k * n_nodes + j,G_i[k * n_nodes + j],t_k, global_timestamps[k - 1]);
        }
    }

    //! in fact, H1 is one dimension, here I make all threads calculating the same thing
    ArrayDouble H1_i = view(H1[i]);
    ArrayDouble H2_i = view(H2[i]);
    ArrayDouble H3_i = view(H3[i]);
    for (ulong k = 1; k != 1 + Total_events + 1; k++) {
        H1_i[global_n[k - 1]] += 1;

        const double t_k = (k != (1 + Total_events) ? global_timestamps[k] : end_time);
        const double ebt = std::exp(-decay * (t_k - global_timestamps[k - 1]));
        H2_i[global_n[k - 1]] -= t_k - global_timestamps[k - 1];

        //! recall that all g_i are same
        //! thread_i calculate H3_i
        H3_i[global_n[k - 1]] -= (1 - ebt) / decay * g_i[(k - 1) * n_nodes + i];
    }
}

ulong ModelHawkesCustom::get_n_coeffs() const {
    //!seems not ever used in this stage
    return n_nodes + n_nodes * n_nodes + n_nodes * MaxN_of_f;
}