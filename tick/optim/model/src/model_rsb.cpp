// License: BSD 3 clause

#include "model_rsb.h"

ModelRsb::ModelRsb(const double _decay, const ulong _MaxN, const int max_n_threads) :
        ModelHawkesFixedKernLogLik(max_n_threads), MaxN(_MaxN), decay(_decay) {}

void ModelRsb::set_data(const SArrayDoublePtrList1D &_timestamps,
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

double ModelRsb::loss(const ArrayDouble &coeffs) {
    const double loss =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelRsb::loss_dim_i,
                                         this,
                                         coeffs);
    return loss / Total_events;
}

double ModelRsb::loss_dim_i(const ulong i, const ArrayDouble &coeffs) {
    double loss = 0;

    const ArrayDouble mu_i = view(coeffs, get_mu_i_first_index(i), get_mu_i_last_index(i));

    //term 1
    for (ulong k = 1; k != n_total_jumps + 1; k++)
        if (type_n[k] == i + 1) {
            double tmp_s = mu_i[global_n[k - 1]];
            if (tmp_s <= 0) {
                printf("debug: dim %d: %f %d %d\n", i, tmp_s, global_n[k-1], k);
                printf("debug: %f %f %f %f %f\n", mu_i[0], mu_i[1], mu_i[2], mu_i[3], mu_i[4]);
                printf("%d\n", Total_events);
                TICK_ERROR("The sum of the influence on someone cannot be negative. "
                                   "Maybe did you forget to add a positive constraint to "
                                   "your proximal operator, in ModelRsb::loss_dim_i");
            }
            loss += log(tmp_s);
        }

    //term 2, 3
    for (ulong k = 1; k != n_total_jumps + 1; k++)
        loss -= mu_i[global_n[k - 1]] * (global_timestamps[k] - global_timestamps[k - 1]);
    loss -= mu_i[global_n[n_total_jumps]] * (end_time - global_timestamps[n_total_jumps]);

    //add a constant to the loss, then inverse the loss to make it convex
    return -end_time - loss;
}

void ModelRsb::grad(const ArrayDouble &coeffs,
                                      ArrayDouble &out) {
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelRsb::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= Total_events;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}

void ModelRsb::grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) {
    const ArrayDouble mu_i = view(coeffs, get_mu_i_first_index(i), get_mu_i_last_index(i));
    ArrayDouble grad_mu_i = view(out, get_mu_i_first_index(i), get_mu_i_last_index(i));

    //! grad of mu_i(n)
    for (ulong k = 1; k < n_total_jumps + 1; ++k) {
        int tmp_flag = 0;
        if (k == n_total_jumps + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            double denominator = mu_i[global_n[k - 1]];
            grad_mu_i[global_n[k - 1]] += 1.0 / denominator;
        }
    }

    for (ulong k = 1; k < 1 + n_total_jumps + 1; k++) {
        const double t_k = (k != (n_total_jumps + 1)) ? global_timestamps[k] : end_time;
        grad_mu_i[global_n[k - 1]] -= (t_k - global_timestamps[k - 1]);
    }
}
