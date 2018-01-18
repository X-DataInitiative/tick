// License: BSD 3 clause


#include "tick/hawkes/model/model_hawkes_utils.h"

TimestampListDescriptor describe_timestamps_list(const SArrayDoublePtrList2D &timestamps_list) {
  // Check the number of realizations
  ulong n_realizations = timestamps_list.size();
  if (n_realizations == 0) {
    TICK_ERROR("You must provide at least one realization");
  }

  // Check the number of nodes
  ulong n_nodes = timestamps_list[0].size();
  if (n_nodes == 0) {
    TICK_ERROR("Your realization should have more than one node");
  }

  auto n_total_jumps_per_realization = VArrayULong::new_ptr(n_realizations);
  n_total_jumps_per_realization->init_to_zero();

  auto n_total_jumps_per_node = SArrayULong::new_ptr(n_nodes);
  n_total_jumps_per_node->init_to_zero();

  for (ulong r = 0; r < n_realizations; ++r) {
    SArrayDoublePtrList1D realization_r = timestamps_list[r];
    if (realization_r.size() != n_nodes) {
      TICK_ERROR("All realizations should have " << n_nodes << " nodes, but realization "
                                                 << r << " has " << realization_r.size()
                                                 << " nodes");
    }

    for (ulong i = 0; i < n_nodes; i++) {
      (*n_total_jumps_per_realization)[r] += realization_r[i]->size();
      (*n_total_jumps_per_node)[i] += realization_r[i]->size();
    }
  }

  TimestampListDescriptor timestamps_list_descriptor{
    n_realizations, n_nodes, n_total_jumps_per_realization, n_total_jumps_per_node
  };
  return timestamps_list_descriptor;
}

TimestampListDescriptor describe_timestamps_list(const SArrayDoublePtrList2D &timestamps_list,
                                                 const VArrayDoublePtr end_times) {
  auto timestamps_list_descriptor = describe_timestamps_list(timestamps_list);

  if (timestamps_list_descriptor.n_realizations != end_times->size()) {
    TICK_ERROR(
      "You must provide as many end_times (" << end_times->size() << ") as realizations ("
                                             << timestamps_list_descriptor.n_realizations << ")");
  }

  for (ulong r = 0; r < timestamps_list_descriptor.n_realizations; r++) {
    SArrayDoublePtrList1D realization_r = timestamps_list[r];
    for (ulong i = 0; i < timestamps_list_descriptor.n_nodes; i++) {
      if (realization_r[i]->size() > 0) {
        double last_time_i = (*realization_r[i])[realization_r[i]->size() - 1];
        if ((*end_times)[r] < last_time_i) {
          TICK_ERROR("Provided end_time (" << (*end_times)[i] << ") is smaller than last time of "
                                           << "component " << i << " (" << last_time_i << ")")
        }
      }
    }
  }

  return timestamps_list_descriptor;
}
