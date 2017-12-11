#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "hawkes_fixed_kern_loglik_list.h"
#include "tick/hawkes/model/hawkes_fixed_sumexpkern_loglik.h"

/** \class ModelHawkesFixedSumExpKernLogLikList
 * \brief Class for computing L2 Contrast function and gradient for Hawkes processes with
 * exponential kernels with fixed exponent (i.e., alpha*beta*e^{-beta t}, with fixed beta)
 * on a list of realizations
 */
class DLL_PUBLIC ModelHawkesFixedSumExpKernLogLikList : public ModelHawkesFixedKernLogLikList {
  //! @brief Value of decays array for this model
  ArrayDouble decays;

 public:
  /**
   * @brief Constructor
   * \param decay : decay for this model (remember that decay is fixed!) 
   * \param max_n_threads : number of cores to be used for multithreading. If negative,
   * the number of physical cores will be used
   */
  ModelHawkesFixedSumExpKernLogLikList(const ArrayDouble &decay,
                                       const int max_n_threads = 1);

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

  std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) override {
    return std::unique_ptr<ModelHawkesFixedSumExpKernLogLik>(
      new ModelHawkesFixedSumExpKernLogLik(decays, n_threads));
  }

  ulong get_n_coeffs() const override;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LOGLIK_LIST_H_
