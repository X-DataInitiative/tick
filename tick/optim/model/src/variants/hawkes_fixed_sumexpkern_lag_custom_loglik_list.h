#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LAG_CUSTOM_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LAG_CUSTOM_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "base.h"
#include "hawkes_fixed_custom_loglik_list.h"
#include "../hawkes_fixed_sumexpkern_lag_loglik_custom.h"

class DLL_PUBLIC ModelHawkesFixedSumExpKernLagCustomLogLikList : public ModelHawkesCustomLogLikList{
  ArrayDouble decays;
  ArrayDouble lags;

  ulong MaxN_of_f;

 public:
ModelHawkesFixedSumExpKernLagCustomLogLikList(const ArrayDouble &decay,
                                             const ArrayDouble &lags,
                                             const ulong _MaxN_of_f,
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

    void set_MaxN_of_f(ulong _MaxN_of_f) {
        MaxN_of_f = _MaxN_of_f;
    }

    ulong get_MaxN_of_f() const {
        return MaxN_of_f;
    }

    std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) {
      return std::unique_ptr<ModelHawkesSumExpCustomLag>(
              new ModelHawkesSumExpCustomLag(decays, lags, MaxN_of_f, n_threads));
    }

    ulong get_n_coeffs() const override;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_SUMEXPKERN_LAG_CUSTOM_LOGLIK_LIST_H_
