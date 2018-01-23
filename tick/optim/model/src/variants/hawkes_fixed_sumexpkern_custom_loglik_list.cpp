// License: BSD 3 clause


#include "hawkes_fixed_sumexpkern_custom_loglik_list.h"
#include "hawkes_utils.h"

ModelHawkesFixedSumExpKernCustomLogLikList::ModelHawkesFixedSumExpKernCustomLogLikList(
  const ArrayDouble &decays, const ulong _MaxN_of_f, const int max_n_threads) :
  ModelHawkesFixedKernLogLikList(max_n_threads), decays(decays), MaxN_of_f(_MaxN_of_f) {}

ulong ModelHawkesFixedSumExpKernCustomLogLikList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays() + n_nodes * MaxN_of_f;
}

void ModelHawkesFixedSumExpKernCustomLogLikList::set_data(const SArrayDoublePtrList2D &timestamps_list,
                                                          const SArrayDoublePtrList1D &global_n_list,
                                                            const VArrayDoublePtr end_times) {

    const auto timestamps_list_descriptor = describe_timestamps_list(timestamps_list, end_times);
    n_realizations = timestamps_list_descriptor.n_realizations;
    set_n_nodes(timestamps_list_descriptor.n_nodes);
    n_jumps_per_node = timestamps_list_descriptor.n_jumps_per_node;
    n_jumps_per_realization = timestamps_list_descriptor.n_jumps_per_realization;

    this->timestamps_list = timestamps_list;
    this->end_times = end_times;

    this->global_n_list = global_n_list;


    weights_computed = false;
}