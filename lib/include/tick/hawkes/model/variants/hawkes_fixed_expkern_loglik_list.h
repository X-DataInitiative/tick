#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "hawkes_fixed_kern_loglik_list.h"
#include "tick/hawkes/model/hawkes_fixed_expkern_loglik.h"

/** \class ModelHawkesFixedExpKernLogLikList
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 * on a list of realizations
 */
class DLL_PUBLIC ModelHawkesFixedExpKernLogLikList : public ModelHawkesFixedKernLogLikList {
  //! @brief Value of decay for this model. Shared by all kernels
  double decay;

 public:
  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!) 
   * \param max_n_threads : number of cores to be used for multithreading. If negative,
   * the number of physical cores will be used
   */
  ModelHawkesFixedExpKernLogLikList(const double decay,
                                    const int max_n_threads = 1);

  /**
   * @brief Set decays and reset weights computing
   * @param decays : new decays to be set
   */
  void set_decay(const double decay) {
    weights_computed = false;
    this->decay = decay;
  }

  double get_decay() const {
    return decay;
  }

  std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) override {
    return std::unique_ptr<ModelHawkesFixedExpKernLogLik>(
      new ModelHawkesFixedExpKernLogLik(decay, n_threads));
  }

  ulong get_n_coeffs() const override;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_LOGLIK_LIST_H_
