//
// Created by Martin Bompaire on 16/07/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_
#define TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_

// License: BSD 3 clause

#include "tick/base/base.h"

#include "base/hawkes_fixed_kern_loglik.h"

class ModelHawkesFixedExpKernLogLikList;

/**
 * \class ModelHawkesFixedExpKernLogLik
 * \brief Class for computing loglikelihood function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., \f$ \alpha \beta e^{-\beta t} \f$, with fixed
 * decay)
 */
class DLL_PUBLIC ModelHawkesFixedExpKernLogLik : public ModelHawkesFixedKernLogLik {
 private:
  //! @brief Value of decay for this model
  double decay;

 public:
  //! @brief Default constructor
  //! @note This constructor is only used to create vectors of ModelHawkesFixedExpKernLeastSq
  ModelHawkesFixedExpKernLogLik() : ModelHawkesFixedKernLogLik() {}

  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!)
   * \param n_threads : number of threads that will be used for parallel computations
   */
  explicit ModelHawkesFixedExpKernLogLik(const double decay, const int max_n_threads = 1);

 private:
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
    return n_nodes + i * n_nodes;
  }

  /**
   * @brief Return the end of alpha i coefficients in a coeffs vector
   * @param i : selected dimension
   */
  ulong get_alpha_i_last_index(const ulong i) const override {
    return n_nodes + (i + 1) * n_nodes;
  }

 public:
  ulong get_n_coeffs() const override;

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

  friend ModelHawkesFixedExpKernLogLikList;
};

#endif  // TICK_OPTIM_MODEL_SRC_HAWKES_FIXED_EXPKERN_LOGLIK_H_
