#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_CUSTOM_TYPE2_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_CUSTOM_TYPE2_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "tick/base/base.h"
#include "hawkes_fixed_custom_loglik_list.h"
#include "hawkes_fixed_sumexpkern_loglik_custom2.h"

class DLL_PUBLIC ModelHawkesFixedSumExpKernCustomType2LogLikList
    : public ModelHawkesCustomLogLikList {
 protected:
  ulong MaxN;
  ArrayDouble decays;

 public:
  ModelHawkesFixedSumExpKernCustomType2LogLikList(const ArrayDouble &decay, const ulong _MaxN,
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

  ulong get_n_decays() const { return decays.size(); }

  void set_MaxN(ulong _MaxN) { MaxN = _MaxN; }

  ulong get_MaxN() const { return MaxN; }

  std::unique_ptr<ModelHawkesLogLikSingle> build_model(const int n_threads) {
    return std::unique_ptr<ModelHawkesSumExpCustomType2>(
        new ModelHawkesSumExpCustomType2(decays, MaxN, n_threads));
  }

  ulong get_n_coeffs() const;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_CUSTOM_TYPE2_LOGLIK_LIST_H_
