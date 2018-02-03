#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_CUSTOM_LOGLIK_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_CUSTOM_LOGLIK_LIST_H_

// License: BSD 3 clause

#include "base.h"
#include "../base/hawkes_list.h"
#include "../base/hawkes_fixed_kern_loglik.h"

class DLL_PUBLIC ModelHawkesCustomLogLikList : public ModelHawkesList {

    SArrayLongPtrList1D global_n_list;

    std::vector<std::unique_ptr<ModelHawkesFixedKernLogLik> > model_list;

public:

    void set_data(const SArrayDoublePtrList2D &timestamps_list, const SArrayLongPtrList1D &global_n_list, const VArrayDoublePtr end_times);

    explicit ModelHawkesCustomLogLikList(const int max_n_threads = 1);

    ModelHawkesCustomLogLikList(const ModelHawkesCustomLogLikList& model) = delete;
    ModelHawkesCustomLogLikList& operator=(const ModelHawkesCustomLogLikList& model) = delete;

    void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

    void compute_weights();

    double loss(const ArrayDouble &coeffs) override;

    double loss_i(const ulong i, const ArrayDouble &coeffs) override;

    void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

    void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

    double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

    ulong get_rand_max() const {
      return get_n_total_jumps();
    }

    virtual ulong get_n_coeffs(){
        TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }
protected:
    virtual std::unique_ptr<ModelHawkesFixedKernLogLik> build_model(const int n_threads) {
      TICK_CLASS_DOES_NOT_IMPLEMENT("");
    }

private:
    std::tuple<ulong, ulong> get_realization_node(ulong i_r);

    void compute_weights_i_r(const ulong i_r);

    double loss_i_r(const ulong i_r, const ArrayDouble &coeffs);

    void grad_i_r(const ulong i_r, ArrayDouble &out, const ArrayDouble &coeffs);

    std::pair<ulong, ulong> sampled_i_to_realization(const ulong sampled_i);
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_CUSTOM_LOGLIK_LIST_H_
