//
// Created by pwu on 12/19/17.
//

#ifndef TICK_HAWKES_CUSTOM_H
#define TICK_HAWKES_CUSTOM_H

#include "hawkes.h"

class Hawkes_custom : public Hawkes {
public:
    //! Peng Wu, An array, indicaitng the global status after the i_th event
    VArrayULong global_n;

    //! @brief the max value of n kept for all f_i(n)
    ulong MaxN_of_f;
    ArrayDoubleList1D f_i;
    ArrayDouble f_i_Max;

//
//    using Hawkes::get_baseline_bound;
//    using Hawkes::baselines;
//    using Hawkes::kernels;

    //! variables used to calclate status

public :
    /**
     * @brief A constructor for an empty multidimensional Hawkes process
     * \param n_nodes : The dimension of the Hawkes process
     */
    explicit Hawkes_custom(unsigned int n_nodes, int seed, ulong _MaxN_of_f, ArrayDoubleList1D _f_i);

    // This forbids the unwanted copy of an Hawkes process
    Hawkes_custom(Hawkes_custom &hawkes_custom) = delete;

    /**
     * @brief Updates the current time so that it goes forward of delay seconds
     * The intensities must be updated and track recorded if needed
     * Returns false if negative intensities were encountered
     * \param delay : Time to update
     * \param intensity : The intensity vector to update
     * \param total_intensity_bound : If not NULL then used to set a bound of
     * total future intensity
     */
    bool update_time_shift_(double delay,
                            ArrayDouble &intensity,
                            double *total_intensity_bound);

    void update_jump(int index);
};


#endif //TICK_HAWKES_CUSTOM_H
