// License: BSD 3 clause


#include "hawkes_fixed_kern_custom_loglik_list.h"
#include "hawkes_utils.h"

ModelHawkesFixedSumExpKernCustomLogLikList::ModelHawkesFixedSumExpKernCustomLogLikList(
  const ArrayDouble &decays, const ulong _MaxN_of_f, const int max_n_threads) :
  ModelHawkesFixedKernLogLikList(max_n_threads), decays(decays), MaxN_of_f(_MaxN_of_f) {}

ulong ModelHawkesFixedCustomLogLikList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays() + n_nodes * (MaxN_of_f - 1);
}

void ModelHawkesFixedCustomLogLikList::set_data(const SArrayDoublePtrList2D &timestamps_list,
                                                          const SArrayLongPtrList1D &global_n_list,
                                                            const VArrayDoublePtr end_times) {

    printf("\nModelHawkesFixedSumExpKernCustomLogLikList::set_data is called.\n");

    const auto timestamps_list_descriptor = describe_timestamps_list(timestamps_list, end_times);
    n_realizations = timestamps_list_descriptor.n_realizations;
    set_n_nodes(timestamps_list_descriptor.n_nodes);
    n_jumps_per_node = timestamps_list_descriptor.n_jumps_per_node;
    n_jumps_per_realization = timestamps_list_descriptor.n_jumps_per_realization;

    this->timestamps_list = timestamps_list;
    this->end_times = end_times;

    this->global_n_list = global_n_list;

    weights_computed = false;

    n_nodes--;
}

void ModelHawkesFixedCustomLogLikList::compute_weights() {
    if (!model_list.empty() && timestamps_list.size() != model_list.size()) {
        TICK_ERROR("Cannot compute weights as timestamps have not been stored. "
                           "Did you use incremental_fit?");
    }
    model_list = std::vector<std::unique_ptr<ModelHawkesFixedKernLogLik> >(n_realizations);

    for (ulong r = 0; r < n_realizations; ++r) {

        printf("ModelHawkesFixedSumExpKernCustomLogLikList::compute_weights loop:%d",r);

        model_list[r] = build_model(1);
        model_list[r]->set_data(timestamps_list[r], (*end_times)[r]);
//        model_list[r]->set_data(timestamps_list[r], global_n_list[r], (*end_times)[r]);
        model_list[r]->allocate_weights();
    }

    parallel_run(get_n_threads(), n_realizations * n_nodes,
                 &ModelHawkesFixedKernLogLikList::compute_weights_i_r, this);

    for (auto &model : model_list) {
        model->weights_computed = true;
    }
    weights_computed = true;
}