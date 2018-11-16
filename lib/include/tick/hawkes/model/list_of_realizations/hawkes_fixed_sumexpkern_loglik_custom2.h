#ifndef TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP_TYPE2
#define TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP_TYPE2

// License: BSD 3 clause

#include "tick/base/base.h"

#include "tick/hawkes/model/base/model_hawkes_loglik.h"

// class ModelHawkesCustomList;

/**
 * \class ModelHawkesCustom
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., \f$ \alpha \beta e^{-\beta t} \f$, with fixed
 * decay)
 */
class DLL_PUBLIC ModelHawkesSumExpCustomType2 : public ModelHawkesLogLikSingle {
 protected:
  //! @brief the max number of states
  ulong MaxN;

  //! Peng Wu, An array, containing timestamps of all type of events, sorted
  ArrayDouble global_timestamps;

  //! Peng Wu, An array, indicating how many n (num of orders) is there AFTER GLOBAL timestamp i
  ArrayLong global_n;

  //! Peng Wu, An array, indicating the type of event of GLOBAL timestamp i
  ArrayULong type_n;

  //! Peng wu, length of the previous two arrays
  ulong Total_events;

  //! @brief Value of decay for this model
  ArrayDouble decays;

 public:
  ModelHawkesSumExpCustomType2(const ArrayDouble &_decays, const ulong _MaxN,
                               const int max_n_threads = 1);

  void set_data(const SArrayDoublePtrList1D &_timestamps, const SArrayLongPtr _global_n,
                const double _end_times);

 private:
  void allocate_weights();

  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  void compute_weights_dim_i(const ulong i);  //! override;

  /**
   * @brief Return the start of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */

  ulong get_mu_i_first_index(const ulong i) const { return MaxN * i; }

  ulong get_mu_i_last_index(const ulong i) const { return MaxN * (i + 1); }

  ulong get_alpha_u_i_j_index(const ulong u, const ulong i, const ulong j) const {
    return n_nodes * MaxN + u * n_nodes * n_nodes + i * n_nodes + j;
  }

 public:
  ulong get_n_coeffs() const override { return n_nodes * MaxN + n_nodes * n_nodes * decays.size(); }

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

  //! override the loss_dim_i and grad_dim_i from src/model_hawkes_loglik.h

  double loss_dim_i(const ulong i, const ArrayDouble &coeffs);

  void grad_dim_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out);

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  double loss(const ArrayDouble &coeffs) override;

  //    friend ModelHawkesCustomList;
};

#endif  // TICK_OPTIM_MODEL_SRC_CUSTOM_SUMEXP_TYPE2
