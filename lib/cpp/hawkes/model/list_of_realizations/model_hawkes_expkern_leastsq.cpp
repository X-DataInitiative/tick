// License: BSD 3 clause

#include "tick/hawkes/model/list_of_realizations/model_hawkes_expkern_leastsq.h"

ModelHawkesExpKernLeastSq::ModelHawkesExpKernLeastSq(
  const SArrayDouble2dPtr decays,
  const int max_n_threads,
  const unsigned int optimization_level)
  : ModelHawkesLeastSq(max_n_threads, optimization_level),
    decays(decays) {
  aggregated_model = std::unique_ptr<ModelHawkesExpKernLeastSqSingle>(
    new ModelHawkesExpKernLeastSqSingle(decays, max_n_threads, optimization_level));
}

void ModelHawkesExpKernLeastSq::hessian(ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  auto *casted_model = static_cast<ModelHawkesExpKernLeastSqSingle *>(aggregated_model.get());
  casted_model->hessian(out);
}

void ModelHawkesExpKernLeastSq::compute_weights_i_r(
  const ulong i_r, std::vector<ModelHawkesExpKernLeastSqSingle> &model_list) {
  const ulong r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;

  model_list[r].compute_weights_i(i);
}

void ModelHawkesExpKernLeastSq::compute_weights_timestamps_list() {
  auto model_list =
    std::vector<ModelHawkesExpKernLeastSqSingle>(n_realizations);

  for (ulong r = 0; r < n_realizations; ++r) {
    model_list[r] = ModelHawkesExpKernLeastSqSingle(decays, 1, optimization_level);
    model_list[r].set_data(timestamps_list[r], (*end_times)[r]);
    model_list[r].allocate_weights();
  }

  // Multithreaded computation of the arrays
  parallel_run(get_n_threads(), n_realizations * n_nodes,
               &ModelHawkesExpKernLeastSq::compute_weights_i_r, this, model_list);

  for (ulong r = 0; r < n_realizations; ++r) {
    Dg.mult_incr(model_list[r].Dg, 1);
    Dg2.mult_incr(model_list[r].Dg2, 1);
    C.mult_incr(model_list[r].C, 1);
    E.mult_incr(model_list[r].E, 1);
  }
}

void ModelHawkesExpKernLeastSq::compute_weights_timestamps(
  const SArrayDoublePtrList1D &timestamps, double end_time) {
  auto model = ModelHawkesExpKernLeastSqSingle(decays, get_n_threads(), optimization_level);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  Dg.mult_incr(model.Dg, 1);
  Dg2.mult_incr(model.Dg2, 1);
  C.mult_incr(model.C, 1);
  E.mult_incr(model.E, 1);
}

void ModelHawkesExpKernLeastSq::allocate_weights() {
  Dg = ArrayDouble2d(n_nodes, n_nodes);
  Dg.init_to_zero();
  Dg2 = ArrayDouble2d(n_nodes, n_nodes);
  Dg2.init_to_zero();
  C = ArrayDouble2d(n_nodes, n_nodes);
  C.init_to_zero();
  E = ArrayDouble2d(n_nodes, n_nodes * n_nodes);
  E.init_to_zero();

  weights_allocated = true;
}

void ModelHawkesExpKernLeastSq::synchronize_aggregated_model() {
  auto *casted_model = static_cast<ModelHawkesExpKernLeastSqSingle *>(aggregated_model.get());

  casted_model->set_n_nodes(n_nodes);
  casted_model->max_n_threads = max_n_threads;

  // We make views to avoid copies
  casted_model->Dg = view(Dg);
  casted_model->Dg2 = view(Dg2);
  casted_model->C = view(C);
  casted_model->E = view(E);
  casted_model->end_time = end_times->sum();

  casted_model->n_total_jumps = n_jumps_per_realization->sum();
  casted_model->n_jumps_per_node = n_jumps_per_node;

  casted_model->weights_computed = weights_computed;
}

ulong ModelHawkesExpKernLeastSq::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes;
}
