#ifndef TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP
#define TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP

// License: BSD 3 clause

#include "tick/base/base.h"

#include "tick/hawkes/model/base/model_hawkes_loglik.h"

/**
 * \class ModelHawkesCustom
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., \f$ \alpha \beta e^{-\beta t} \f$, with fixed
 * decay)
 */
class DLL_PUBLIC ModelHawkesSumExpCustom : public ModelHawkesLogLikSingle {
 protected:
  //! @brief the max value of n kept for all f_i(n)
  ulong MaxN_of_f;
  //! Peng wu, length of the previous two arrays
  ulong Total_events;

  //! @brief Value of decay for this model
  ArrayDouble decays;

  // Some arrays used for intermediate computings. They are initialized in init()

  //! Peng Wu, An array, containing timestamps of all type of events, sorted
  ArrayDouble global_timestamps;

  //! Peng Wu, An array, indicating how many n (num of orders) is there AFTER GLOBAL timestamp i
  ArrayLong global_n;

  //! Peng Wu, An array, indicating the type of event of GLOBAL timestamp i
  ArrayULong type_n;

  //! @brief auxiliary variable H1 described in the document
  ArrayDoubleList1D H1;

  //! @brief auxiliary variable H2 described in the document
  ArrayDoubleList1D H2;

  //! @brief auxiliary variable H3 described in the document
  // H3 experimental here
  ArrayDoubleList1D H3;

 public:
  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!)
   * \param n_threads : number of threads that will be used for parallel computations
   */
  ModelHawkesSumExpCustom(const ArrayDouble _decays, const ulong _MaxN_of_f,
                          const int max_n_threads = 1);

  void set_data(const SArrayDoublePtrList1D &_timestamps, const SArrayLongPtr _global_n,
                const double _end_times);

 private:
  void allocate_weights();

  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  void compute_weights_dim_i(const ulong i);  //! ;

  //! you know what it is
  ulong get_alpha_u_i_j_index(const ulong u, const ulong i, const ulong j) const {
    return n_nodes + u * n_nodes * n_nodes + i * n_nodes + j;
  }

 public:
  ulong get_n_coeffs() const {
    return n_nodes + n_nodes * n_nodes * decays.size() + n_nodes * (MaxN_of_f);
  }

  ulong get_n_decays() const { return decays.size(); }

  //! @brief Returns decay that was set
  SArrayDoublePtr get_decays() const {
    ArrayDouble copied_decays = decays;
    return copied_decays.as_sarray_ptr();
  }

  /**
   * @brief Set new decay
   * \param decay : new decay
   * \note Weights will need to be recomputed
   */
  void set_decays(ArrayDouble decays) {
    this->decays = decays;
    weights_computed = false;
  }

  //!  the loss_dim_i and grad_dim_i from src/model_hawkes_loglik.h

  double loss_dim_i(const ulong i, const ArrayDouble &coeffs);

  void grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

  void grad(const ArrayDouble &coeffs, ArrayDouble &out);

  double loss(const ArrayDouble &coeffs);
  //    friend ModelHawkesCustomList;
};

#endif  // TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP