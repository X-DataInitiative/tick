//
// Created by pwu on 12/19/17.
//

#include "hawkes_custom2.h"


Hawkes_customType2::Hawkes_customType2(unsigned int n_nodes, int seed, ulong _MaxN, const SArrayDoublePtrList1D &_mu_)
        : Hawkes(n_nodes, seed), global_n(0) {
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
    return flag_negative_intensity1;
}

void Hawkes_customType2::update_jump(int index) {
    last_global_n = (unsigned long) std::rand() % MaxN;
    global_n.append1(last_global_n);
    // We make the jump on the corresponding signal
    timestamps[index]->append1(time);
    n_total_jumps++;
}

void Hawkes_customType2::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound) {
    *total_intensity_bound = 0;
    for (unsigned int i = 0; i < n_nodes; i++) {
        intensity[i] = mu_[i]->operator[](last_global_n);
        *total_intensity_bound += mu_Max[i];
    }
}