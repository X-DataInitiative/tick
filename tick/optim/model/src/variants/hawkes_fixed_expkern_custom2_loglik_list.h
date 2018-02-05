#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_CUSTOM2_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_CUSTOM2_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "base.h"
#include "hawkes_fixed_custom_loglik_list.h"
#include "../hawkes_fixed_expkern_loglik_custom2.h"

class DLL_PUBLIC ModelHawkesFixedExpKernCustomType2LogLikList : public ModelHawkesCustomLogLikList{
    double decay;

  ulong MaxN;

 public:
ModelHawkesFixedExpKernCustomType2LogLikList(const double &_decay,
                                             const ulong _MaxN,
                                             const int max_n_threads = 1);

  double get_decay() const {
    return this->decay;
  }

  void set_decay(double &_decay) {
    this->decay = _decay;
    weights_computed = false;
  }

    void set_MaxN(ulong _MaxN) {
        MaxN = _MaxN;
    }

    ulong get_MaxN() const {
        return MaxN;
    }

    std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) {
      return std::unique_ptr<ModelHawkesCustomType2>(
              new ModelHawkesCustomType2(decay, MaxN, n_threads));
    }

    ulong get_n_coeffs() const override;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_FIXED_EXPKERN_CUSTOM2_LOGLIK_LIST_H_
