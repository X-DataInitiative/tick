//
// Created by pwu on 12/19/17.
//

#include "hawkes_custom2.h"

Hawkes_customType2::Hawkes_customType2(unsigned int n_nodes, int seed, ulong _MaxN, const SArrayDoublePtrList1D &_mu_, const ArrayDouble &extrainfo, const std::string _simu_mode)
        : Hawkes(n_nodes, seed), global_n(0), Qty(0), simu_mode(_simu_mode){
    this->MaxN = _MaxN;
    this->mu_ = _mu_;

    mu_Max = ArrayDouble(n_nodes);
    for (ulong i = 0; i != n_nodes; ++i)
        mu_Max[i] = mu_[i]->max();
    last_global_n = 0;

    if(simu_mode == "random")
        return;
    else if(simu_mode == "generate"){
        current_num = extrainfo[0];
        avg = extrainfo[1];
        avg_order_size = ArrayDouble(n_nodes);
        for(ulong k = 0; k != n_nodes; ++k)
            avg_order_size[k] = extrainfo[2 + k];
    }
    else
        TICK_ERROR("Unknown scenario");
}

Hawkes_customType2::Hawkes_customType2(unsigned int n_nodes, int seed, ulong _MaxN, const SArrayDoublePtrList1D &_mu_)
        : Hawkes(n_nodes, seed), global_n(0), Qty(0), simu_mode("random"){
    this->MaxN = _MaxN;
    this->mu_ = _mu_;

    mu_Max = ArrayDouble(n_nodes);
    for (ulong i = 0; i != n_nodes; ++i)
        mu_Max[i] = mu_[i]->max();
    last_global_n = 0;
}

bool Hawkes_customType2::update_time_shift_(double delay,
                                       ArrayDouble &intensity,
                                       double *total_intensity_bound1) {
    if (total_intensity_bound1) *total_intensity_bound1 = 0;
    bool flag_negative_intensity1 = false;

    // We loop on the contributions
    for (unsigned int i = 0; i < n_nodes; i++) {
        bool flag = false;
        if(simu_mode == "random")
            flag = true;
        else if(simu_mode == "generate")
            if(current_num > 0 || (current_num + avg_order_size[i] > 0))
                flag = true;
        if (flag) {
            intensity[i] = mu_[i]->operator[](last_global_n);
            if (total_intensity_bound1)
                *total_intensity_bound1 += mu_Max[i];

            for (unsigned int j = 0; j < n_nodes; j++) {
                HawkesKernelPtr &k = kernels[i * n_nodes + j];

                if (k->get_support() == 0) continue;
                double bound = 0;
                intensity[i] += k->get_convolution(get_time() + delay, *timestamps[j], &bound);

                if (total_intensity_bound1) {
                    *total_intensity_bound1 += bound;
                }
                if (intensity[i] < 0) {
                    if (threshold_negative_intensity) intensity[i] = 0;
                    flag_negative_intensity1 = true;
                }
            }
        }
        else
            intensity[i] = 0;
    }
    return flag_negative_intensity1;
}

void Hawkes_customType2::update_jump(int index) {
    if (simu_mode == "random") {
        last_global_n = (unsigned long) std::rand() % MaxN;
        global_n.append1(last_global_n);
        // We make the jump on the corresponding signal
        timestamps[index]->append1(time);
        n_total_jumps++;
    }
    else if (simu_mode == "generate") {
        current_num += avg_order_size[index];
        double exact = current_num / avg;
        ulong round = floor(exact + 0.5);  //round a number
        last_global_n = (round > MaxN - 1) ? MaxN - 1 : round;
        if(current_num < 1.5 * avg)
            last_global_n = 1;
        if(current_num == 0)
            last_global_n = 0;

        global_n.append1(last_global_n);
        Qty.append1(current_num);
        // We make the jump on the corresponding signal
        timestamps[index]->append1(time);
        n_total_jumps++;
    }
    else {
        TICK_ERROR("Unknown simulation scenario.");
    }
}

void Hawkes_customType2::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound) {
    *total_intensity_bound = 0;
    for (unsigned int i = 0; i < n_nodes; i++) {
        intensity[i] = mu_[i]->operator[](last_global_n);
        *total_intensity_bound += mu_Max[i];
    }
}