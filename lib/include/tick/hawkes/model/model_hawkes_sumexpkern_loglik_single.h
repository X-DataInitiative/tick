
#ifndef LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LOGLIK_SINGLE_H_
#define LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LOGLIK_SINGLE_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include "tick/hawkes/model/base/model_hawkes_loglik_single.h"

class ModelHawkesSumExpKernLogLik;

/**
 * \class ModelHawkesSumExpKernLogLikSingle
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * sum exponential kernels with fixed exponent
 * (i.e., \f$ \sum_u \alpha_u \beta_u e^{-\beta_u t} \f$, with fixed decays)
 */
class DLL_PUBLIC ModelHawkesSumExpKernLogLikSingle : public ModelHawkesLogLikSingle {
 private:
  //! @brief Value of decays array for this model
  ArrayDouble decays;

 public:
  //! @brief Default constructor
  //! @note This constructor is only used to create vectors of ModelHawkesSumExpKernLogLikSingle
  ModelHawkesSumExpKernLogLikSingle();

  /**
   * @brief Constructor
   * \param decays : decays for this model (remember that decay is fixed!)
   * \param n_threads : number of threads that will be used for parallel computations
   */
  explicit ModelHawkesSumExpKernLogLikSingle(const ArrayDouble &decays,
                                             const int max_n_threads = 1);

 protected:
  void allocate_weights() override;

  /**
   * @brief Precomputations of intermediate values for component i
   * \param i : selected component
   */
  void compute_weights_dim_i(const ulong i) override;

  /**
   * @brief Return the start of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */
  ulong get_alpha_i_first_index(const ulong i) const override {
    return n_nodes + i * n_nodes * get_n_decays();
  }

  /**
   * @brief Return the end of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */
  ulong get_alpha_i_last_index(const ulong i) const override {
    return n_nodes + (i + 1) * n_nodes * get_n_decays();
  }

 public:
  ulong get_n_coeffs() const override;

  //! @brief Returns decay that was set
  SArrayDoublePtr get_decays() const {
    ArrayDouble copied_decays = decays;
    return copied_decays.as_sarray_ptr();
  }

  /**
   * @brief Set new decays
   * \param decay : new decays
   * \note Weights will need to be recomputed
   */
  void set_decays(ArrayDouble &decays) {
    this->decays = decays;
    weights_computed = false;
  }

  ulong get_n_decays() const {
    return decays.size();
  }

  friend ModelHawkesSumExpKernLogLik;
};

#endif  // LIB_INCLUDE_TICK_HAWKES_MODEL_MODEL_HAWKES_SUMEXPKERN_LOGLIK_SINGLE_H_
