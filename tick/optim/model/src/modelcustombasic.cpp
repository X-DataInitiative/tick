// License: BSD 3 clause


#include "modelcustombasic.h"

ModelCustomBasic::ModelCustomBasic(const double _decay, const ulong _MaxN_of_f, const int max_n_threads) :
        MaxN_of_f(_MaxN_of_f), ModelHawkesFixedKernLogLik(max_n_threads), decay(_decay) {}

void ModelCustomBasic::allocate_weights() {
    if (n_nodes == 0) {
        TICK_ERROR("Please provide valid timestamps before allocating weights")
    }

    Total_events = n_total_jumps - (*n_jumps_per_node)[n_nodes];

    H1 = ArrayDoubleList1D(n_nodes);
    H2 = ArrayDoubleList1D(n_nodes);

    for (ulong i = 0; i != n_nodes; i++) {
        //0 + events + T

        H1[i] = ArrayDouble(MaxN_of_f);
        H1[i].init_to_zero();
        H2[i] = ArrayDouble(MaxN_of_f);
        H2[i].init_to_zero();
    }
}

void ModelCustomBasic::compute_weights_dim_i(const ulong i) {
    //! in fact, H1, H2 is one dimension, here I make all threads calculate the same thing
    ArrayDouble H1_i = view(H1[i]);
    ArrayDouble H2_i = view(H2[i]);
    for (ulong k = 1; k != 1 + n_total_jumps + 1; k++) {
        if(k != (1 + n_total_jumps))
            H1_i[global_n[k - 1]] += (type_n[k] == i + 1 ? 1 : 0);

        const double t_k = (k != (1 + n_total_jumps) ? global_timestamps[k] : end_time);
        H2_i[global_n[k - 1]] -= t_k - global_timestamps[k - 1];
    }
}

void ModelCustomBasic::set_data(const SArrayDoublePtrList1D &_timestamps,
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

double ModelCustomBasic::loss(const ArrayDouble &coeffs) {
    if (!weights_computed) compute_weights();

    const double loss =
            parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                         &ModelCustomBasic::loss_dim_i,
                                         this,
                                         coeffs);
    return loss / Total_events;
}

double ModelCustomBasic::loss_dim_i(const ulong i,
                                              const ArrayDouble &coeffs) {
    const double mu_i = coeffs[i];

//    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));

    ArrayDouble f_i(MaxN_of_f);
    f_i[0] = 1;
    for(ulong k = 1; k != MaxN_of_f; ++k)
        f_i[k] = coeffs[n_nodes + i * (MaxN_of_f - 1)+ k - 1];

    //term 1
    //end_time is T
    double loss = 0;
    for (ulong k = 1; k != n_total_jumps + 1; k++)
        //! insert event t0 = 0
        if (type_n[k] == i + 1)
            loss += log(f_i[global_n[k - 1]]);

    //term 2
    for (ulong k = 1; k != n_total_jumps + 1; k++)
        if (type_n[k] == i + 1) {
            double tmp_s = mu_i;
            if (tmp_s <= 0) {
                TICK_ERROR("The sum of the influence on someone cannot be negative. "
                                   "Maybe did you forget to add a positive constraint to "
                                   "your proximal operator, in ModelCustomBasic::loss_dim_i");
            }
            loss += log(tmp_s);
        }

    //term 3,4
    for (ulong k = 1; k != n_total_jumps + 1; k++)
        loss -= mu_i * (global_timestamps[k] - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    loss -= mu_i * (end_time - global_timestamps[n_total_jumps]) * f_i[global_n[n_total_jumps]];

    //add a constant to the loss, then inverse the loss to make it convex
    return -end_time - loss;
}

void ModelCustomBasic::grad(const ArrayDouble &coeffs,
                                      ArrayDouble &out) {
    if (!weights_computed) compute_weights();
    out.fill(0);

    // This allows to run in a multithreaded environment the computation of each component
    parallel_run(get_n_threads(),
                 n_nodes,
                 &ModelCustomBasic::grad_dim_i,
                 this,
                 coeffs,
                 out);
    out /= Total_events;

    for (ulong k = 0; k != get_n_coeffs(); ++k)
        out[k] = -out[k];
}

void ModelCustomBasic::grad_dim_i(const ulong i,
                                  const ArrayDouble &coeffs,
                                  ArrayDouble &out) {

    const double mu_i = coeffs[i];
    double &grad_mu_i = out[i];

//    const ArrayDouble f_i = view(coeffs, get_f_i_first_index(i), get_f_i_last_index(i));
    ArrayDouble f_i(MaxN_of_f);
    f_i[0] = 1;
    for(ulong k = 1; k != MaxN_of_f; ++k)
        f_i[k] = coeffs[n_nodes + i * (MaxN_of_f - 1)+ k - 1];

//    ArrayDouble grad_f_i = view(out, get_f_i_first_index(i), get_f_i_last_index(i));
    ArrayDouble grad_f_i(MaxN_of_f);

    //! grad of mu_i
    grad_mu_i = 0;
    for (ulong k = 1; k < n_total_jumps + 1; ++k) {
        int tmp_flag = 0;
        if (k == n_total_jumps + 1)
            tmp_flag = 1;
        else if (type_n[k] == i + 1)
            tmp_flag = 1;
        if (tmp_flag) {
            double numerator = 1;
            double denominator = mu_i;
            grad_mu_i += numerator / denominator;
        }
    }

    for (ulong k = 1; k < 1 + n_total_jumps + 1; k++) {
        const double t_k = (k != (n_total_jumps + 1)) ? global_timestamps[k] : end_time;
        grad_mu_i -= (t_k - global_timestamps[k - 1]) * f_i[global_n[k - 1]];
    }

    //! grad of f^i_n
    //! in fact, H1_i for different i keep the same information, same thing for H2, H3
    const ArrayDouble H1_i = view(H1[i]);
    const ArrayDouble H2_i = view(H2[i]);
    for (ulong n = 0; n != MaxN_of_f; ++n) {
        grad_f_i[n] = H1_i[n] / f_i[n] + mu_i * H2_i[n];
    }

    for(ulong k = 1; k != MaxN_of_f; ++k)
        out[n_nodes + i * (MaxN_of_f - 1) + k - 1] = grad_f_i[k];
}