#ifndef TICK_OPTIM_MODEL_SRC_MODELRSB
#define TICK_OPTIM_MODEL_SRC_MODELRSB

// License: BSD 3 clause

#include "base.h"

#include <memory>

#include "base/hawkes_fixed_kern_loglik.h"

/**
 * \class ModelRsb
 */
class DLL_PUBLIC ModelRsb : public ModelHawkesFixedKernLogLik {
public:
    //! Peng Wu, An array, containing timestamps of all type of events, sorted
    ArrayDouble global_timestamps;

    //! Peng Wu, An array, indicating how many n (num of orders) is there AFTER GLOBAL timestamp i
    ArrayLong global_n;

    //! Peng Wu, An array, indicating the type of event of GLOBAL timestamp i
    ArrayULong type_n;

    //! Peng wu, length of the previous two arrays
    ulong Total_events;

    //! @brief the max number of states
    ulong MaxN;

    //! @brief Value of decay for this model
    double decay;

public:
    //! @brief Default constructor
    //! @note This constructor is only used to create vectors of ModelHawkesFixedExpKernLeastSq
    ModelRsb() : ModelHawkesFixedKernLogLik(0), decay(0) {}

    /**
     * @brief Constructor
     * \param decay : decay for this model (remember that decay is fixed!)
     * \param n_threads : number of threads that will be used for parallel computations
     */
    ModelRsb(const double _decay, const ulong _MaxN, const int max_n_threads = 1);

    using ModelHawkesSingle::set_data;
    void set_data(const SArrayDoublePtrList1D &_timestamps,
                  const SArrayLongPtr _global_n,
                  const double _end_times);

private:

    ulong get_mu_i_first_index(const ulong i) const{
        return MaxN * i;
    }

    ulong get_mu_i_last_index(const ulong i) const{
        return MaxN * (i + 1);
    }

public:
    ulong get_n_coeffs() const override{
        return n_nodes * MaxN;
    }

    //! @brief Returns decay that was set
    double get_decay() const {
        return decay;
    }

    /**
     * @brief Set new decay
     * \param decay : new decay
     * \note Weights will need to be recomputed
     */
    void set_decay(double decay) {
        this->decay = decay;
        weights_computed = false;
    }

    //!override the loss_dim_i and grad_dim_i from src/hawkes_fixed_kern_loglik.h

    double loss_dim_i(const ulong i, const ArrayDouble &coeffs) override;

    void grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

    void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

    double loss(const ArrayDouble &coeffs) override;

//    friend ModelHawkesCustomList;
};

#endif  // TICK_OPTIM_MODEL_SRC_MODELRSB
