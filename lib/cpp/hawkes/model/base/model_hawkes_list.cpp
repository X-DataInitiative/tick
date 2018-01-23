// License: BSD 3 clause

#include "tick/hawkes/model/base/model_hawkes_list.h"
#include "tick/hawkes/model/model_hawkes_utils.h"

ModelHawkesList::ModelHawkesList(
  const int max_n_threads,
  const unsigned int optimization_level)
  : ModelHawkes(max_n_threads, optimization_level), n_realizations(0), timestamps_list(0) {
  n_jumps_per_realization = VArrayULong::new_ptr(n_realizations);
  end_times = VArrayDouble::new_ptr(n_realizations);
}

void ModelHawkesList::set_data(const SArrayDoublePtrList2D &timestamps_list,
                               const VArrayDoublePtr end_times) {
  const auto timestamps_list_descriptor = describe_timestamps_list(timestamps_list, end_times);
  n_realizations = timestamps_list_descriptor.n_realizations;
  set_n_nodes(timestamps_list_descriptor.n_nodes);
  n_jumps_per_node = timestamps_list_descriptor.n_jumps_per_node;
  n_jumps_per_realization = timestamps_list_descriptor.n_jumps_per_realization;

  this->timestamps_list = timestamps_list;
  this->end_times = end_times;

  weights_computed = false;
}

unsigned int ModelHawkesList::get_n_threads() const {
  return std::min(this->max_n_threads, static_cast<unsigned int>(n_nodes * n_realizations));
}
