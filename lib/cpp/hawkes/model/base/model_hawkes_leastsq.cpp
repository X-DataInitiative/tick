// License: BSD 3 clause

#include "tick/hawkes/model/base/model_hawkes_leastsq.h"

ModelHawkesLeastSq::ModelHawkesLeastSq(
  const int max_n_threads,
  const unsigned int optimization_level)
  : ModelHawkesList(max_n_threads, optimization_level),
    weights_allocated(false) {}

void ModelHawkesLeastSq::grad_i(const ulong i, const ArrayDouble &coeffs,
                                ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  aggregated_model->grad_i(i, coeffs, out);
}

void ModelHawkesLeastSq::grad(const ArrayDouble &coeffs, ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  aggregated_model->grad(coeffs, out);
}

double ModelHawkesLeastSq::loss_i(const ulong i, const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  return aggregated_model->loss_i(i, coeffs);
}

double ModelHawkesLeastSq::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  return aggregated_model->loss(coeffs);
}

// Full initialization of the arrays H, Dg, Dg2 and C
// Must be performed just once
void ModelHawkesLeastSq::compute_weights() {
  allocate_weights();

  compute_weights_timestamps_list();

  weights_computed = true;
  synchronize_aggregated_model();
}

void ModelHawkesLeastSq::incremental_set_data(const SArrayDoublePtrList1D &timestamps,
                                              double end_time) {
  weights_computed = false;
  if (!weights_allocated) {
    set_n_nodes(timestamps.size());

    allocate_weights();
    n_realizations = 0;
    end_times = VArrayDouble::new_ptr(0);
    n_jumps_per_realization = VArrayULong::new_ptr(0);
    n_jumps_per_node = SArrayULong::new_ptr(n_nodes);
    n_jumps_per_node->init_to_zero();
  } else {
    if (n_nodes != timestamps.size()) {
      TICK_ERROR("Your realization should have " << n_nodes << " nodes but has "
                                                 << timestamps.size() << ".");
    }
  }

  n_realizations += 1;
  end_times->append1(end_time);

  ulong n_total_jumps = 0;
  for (ulong i = 0; i < n_nodes; ++i) {
    n_total_jumps += timestamps[i]->size();
    (*n_jumps_per_node)[i] += timestamps[i]->size();
  }
  n_jumps_per_realization->append1(n_total_jumps);

  compute_weights_timestamps(timestamps, end_time);

  weights_computed = true;
  synchronize_aggregated_model();
}
