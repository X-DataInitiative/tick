#ifndef TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_QRH2_LEASTSQ_LIST_H_
#define TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_QRH2_LEASTSQ_LIST_H_

// License: BSD 3 clause

#include "base.h"
#include "../base/hawkes_list.h"
#include "../hawkes_fixed_sumexpkern_leastsq_qrh2.h"

class DLL_PUBLIC ModelHawkesFixedSumExpKernLeastSqQRH2List : public ModelHawkesList {

    ArrayDouble decays;
    ulong MaxN;

    SArrayLongPtrList1D global_n_list;

    std::vector<std::unique_ptr<ModelHawkesFixedSumExpKernLeastSqQRH2> > model_list;

    //Total valid jumps, in all realizations
    ulong Total_events;

public:

    void set_data(const SArrayDoublePtrList2D &timestamps_list, const SArrayLongPtrList1D &global_n_list, const VArrayDoublePtr end_times);

    ModelHawkesFixedSumExpKernLeastSqQRH2List(const ArrayDouble &decays, const ulong _MaxN, const int max_n_threads);

    explicit ModelHawkesFixedSumExpKernLeastSqQRH2List(const int max_n_threads = 1);

    ModelHawkesFixedSumExpKernLeastSqQRH2List(const ModelHawkesFixedSumExpKernLeastSqQRH2List& model) = delete;
    ModelHawkesFixedSumExpKernLeastSqQRH2List& operator=(const ModelHawkesFixedSumExpKernLeastSqQRH2List& model) = delete;

    void incremental_set_data(const SArrayDoublePtrList1D &timestamps, double end_time);

    void compute_weights();

    double loss(const ArrayDouble &coeffs) override;

    double loss_i(const ulong i, const ArrayDouble &coeffs) override;

    void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

    void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

    double loss_and_grad(const ArrayDouble &coeffs, ArrayDouble &out);

    ulong get_n_coeffs() const override;

protected:
    virtual std::unique_ptr<ModelHawkesFixedSumExpKernLeastSqQRH2> build_model(const int n_threads) {
        return std::unique_ptr<ModelHawkesFixedSumExpKernLeastSqQRH2>(
                new ModelHawkesFixedSumExpKernLeastSqQRH2(decays, MaxN, n_threads, 0));
        //last parameter is the optimization level
    }

private:
    std::tuple<ulong, ulong> get_realization_node(ulong i_r);

    void compute_weights_i_r(const ulong i_r);

    void compute_weights_H_j_r(const ulong j_r);

    double loss_i_r(const ulong i_r, const ArrayDouble &coeffs);

    void grad_i_r(const ulong i_r, ArrayDouble &out, const ArrayDouble &coeffs);

    std::pair<ulong, ulong> sampled_i_to_realization(const ulong sampled_i);
};

#endif  // TICK_OPTIM_MODEL_SRC_VARIANTS_HAWKES_QRH2_LEASTSQ_LIST_H_
