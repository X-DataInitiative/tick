//
// Created by pwu on 1/15/18.
//

#ifndef TICK_HAWKES_FIXED_SUMEXPKERN_LOGLIK_CUSTOM_H
#define TICK_HAWKES_FIXED_SUMEXPKERN_LOGLIK_CUSTOM_H


// License: BSD 3 clause


#include "base.h"

#include "base/hawkes_single.h"

class DLL_PUBLIC ModelHawkesFixedSumExpKernCustom : public ModelHawkesSingle {
public:
    //! @brief Value of decay for this model
    ArrayDouble decays;
protected:
    // Some arrays used for intermediate computings. They are initialized in init()
    //! @brief kernel intensity of node j on node i at time t_i_k
    ArrayDouble2dList1D g;

    //! @brief compensator of kernel intensity of node j on node i between t_i_k and t_i_(k-1)
    ArrayDouble2dList1D G;

    //! @brief compensator of kernel intensity of node j on node i between 0 and end_time
    //! in this custom setting, the sum_G is meanlingless
    ArrayDoubleList1D sum_G;

    //! Peng Wu, An array, containing timestamps of all type of events, sorted
    ArrayDouble global_timestamps;

    //! Peng Wu, An array, indicating how many n (num of orders) is there AFTER GLOBAL timestamp i
    ArrayLong global_n;

    //! Peng Wu, An array, indicating the type of event of GLOBAL timestamp i
    ArrayULong type_n;

    //! Peng wu, length of the previous two arrays
    ulong Total_events;

    //! @brief auxiliary variable H1 described in the document
    ArrayDoubleList1D H1;

    //! @brief auxiliary variable H2 described in the document
    ArrayDoubleList1D H2;

    //! @brief auxiliary variable H3 described in the document
    // H3 experimental here
    ArrayDoubleList1D H3;

    //! @brief the max value of n kept for all f_i(n)
    ulong MaxN_of_f;

public:
//    using ModelHawkesSingle::set_data;
    /**
     * @brief Constructor
     * \param n_threads : number of threads that will be used for parallel computations
     */
    ModelHawkesFixedSumExpKernCustom(const ulong _MaxN_of_f, const ArrayDouble _decays, const int max_n_threads = 1);

    /**
     * @brief Precomputations of intermediate values
     * They will be used to compute faster loss, gradient and hessian norm.
     * \note This computation will be needed again if user modifies decay afterwards.
     */
    void compute_weights();

//    void set_data(const SArrayDoublePtrList2D &timestamps_list,
//                  VArrayDoublePtr end_times);

    /**
     * @brief Compute loss
     * \param coeffs : Point in which loss is computed
     * \return Loss' value
     */
    double loss(const ArrayDouble &coeffs) override;

    /**
     * @brief Compute gradient
     * \param coeffs : Point in which gradient is computed
     * \param out : Array in which the value of the gradient is stored
     */
    void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

protected:
    virtual void allocate_weights();

    /**
     * @brief Precomputations of intermediate values for component i
     * \param i : selected component
     */
    virtual void compute_weights_dim_i(const ulong i);

    /**
     * @brief Compute loss corresponding to component i
     * \param i : selected component
     * \param coeffs : Point in which loss is computed
     * \param out : Array which the result of the gradient will be added to
     * \return Loss' value
     * \note For two different values of i, this function will modify different coordinates of
     * out. Hence, it is thread safe.
     */
    double loss_dim_i(const ulong i, const ArrayDouble &coeffs);

    /**
     * @brief Compute gradient corresponding to component i
     * \param i : selected component
     * \param coeffs : Point in which gradient is computed
     * \param out : Array which the result of the gradient will be added to
     * \note For two different values of i, this function will modify different coordinates of
     * out. Hence, it is thread safe.
     */
    void grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

    /**
     * @brief Return the start of alpha i coefficients in a coeffs vector
     * @param i : selected dimension
     */
    virtual ulong get_alpha_i_first_index(const ulong u, const ulong i) const {
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }

    /**
     * @brief Return the end of alpha i coefficients in a coeffs vector
     * @param i : selected dimension
     */
    virtual ulong get_alpha_i_last_index(const ulong u, const ulong i) const {
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }

    /**
     * @brief Return the start of f_i_n coefficients in a coeffs vector
     * @param i : selected dimension
     */
    virtual ulong get_f_i_first_index(const ulong i) const {
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }

    /**
     * @brief Return the end of f_i_n coefficients in a coeffs vector
     * @param i : selected dimension
     */
    virtual ulong get_f_i_last_index(const ulong i) const {
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }

    virtual ulong get_alpha_u_i_j_index(const ulong u, const ulong i, const ulong j) const {
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }



public:
    //! @brief Returns max of the range of feasible grad_i and loss_i (total number of timestamps)
    inline ulong get_rand_max() const {
        return n_total_jumps;
    }

};


#endif //TICK_HAWKES_FIXED_KERN_LOGLIK_CUSTOM_H
