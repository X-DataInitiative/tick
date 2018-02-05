#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_CUSTOM_BASIC_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_CUSTOM_BASIC_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "base.h"
#include "hawkes_fixed_custom_loglik_list.h"
#include "../modelcustombasic.h"

class DLL_PUBLIC ModelCustomBasicLogLikList : public ModelHawkesCustomLogLikList{
    double decay;

  ulong MaxN_of_f;

 public:
    ModelCustomBasicLogLikList(const double &_decay,
                                             const ulong _MaxN_of_f,
                                             const int max_n_threads = 1);

  double get_decay() const {
    return this->decay;
  }

  void set_decay(double &_decay) {
    this->decay = _decay;
    weights_computed = false;
  }

    void set_MaxN_of_f(ulong _MaxN_of_f) {
        MaxN_of_f = _MaxN_of_f;
    }

    ulong get_MaxN_of_f() const {
        return MaxN_of_f;
    }

    std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) {
      return std::unique_ptr<ModelCustomBasic>(
              new ModelCustomBasic(decay, MaxN_of_f, n_threads));
    }

    ulong get_n_coeffs() const override;
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_CUSTOM_BASIC_LOGLIK_LIST_H_
