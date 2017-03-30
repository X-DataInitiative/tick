
#include "hawkes_fixed_sumexpkern_leastsq_list.h"
#include "../hawkes_utils.h"

ModelHawkesFixedSumExpKernLeastSqList::ModelHawkesFixedSumExpKernLeastSqList(
    const ArrayDouble &decays,
    const unsigned int max_n_threads,
    const unsigned int optimization_level)
    : ModelHawkesLeastSqList(max_n_threads, optimization_level),
      decays(decays), n_decays(decays.size()) {
  aggregated_model = std::unique_ptr<ModelHawkesFixedSumExpKernLeastSq>(
      new ModelHawkesFixedSumExpKernLeastSq(decays, max_n_threads, optimization_level));
}

void ModelHawkesFixedSumExpKernLeastSqList::compute_weights_i_r(
    const ulong i_r, std::vector<ModelHawkesFixedSumExpKernLeastSq> &model_list) {
  const ulong r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;

  model_list[r].compute_weights_i(i);
}

void ModelHawkesFixedSumExpKernLeastSqList::compute_weights_timestamps_list() {
  auto model_list =
      std::vector<ModelHawkesFixedSumExpKernLeastSq>(n_realizations);

  for (ulong r = 0; r < n_realizations; ++r) {
    model_list[r] = ModelHawkesFixedSumExpKernLeastSq(decays, 1, optimization_level);
    model_list[r].set_data(timestamps_list[r], (*end_times)[r]);
    model_list[r].allocate_weights();
  }

  // Multithreaded computation of the arrays
  parallel_run(get_n_threads(), n_realizations * n_nodes,
               &ModelHawkesFixedSumExpKernLeastSqList::compute_weights_i_r, this, model_list);

  for (ulong r = 0; r < n_realizations; ++r) {
    for (ulong i = 0; i < n_nodes; ++i) {
      Dg[i].mult_incr(model_list[r].Dg[i], 1);
      Dgg[i].mult_incr(model_list[r].Dgg[i], 1);
      C[i].mult_incr(model_list[r].C[i], 1);
      E[i].mult_incr(model_list[r].E[i], 1);
    }
  }
}

void ModelHawkesFixedSumExpKernLeastSqList::compute_weights_timestamps(
    const SArrayDoublePtrList1D &timestamps, double end_time) {
  auto model = ModelHawkesFixedSumExpKernLeastSq(decays, get_n_threads(), optimization_level);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  for (ulong i = 0; i < n_nodes; ++i) {
    Dg[i].mult_incr(model.Dg[i], 1);
    Dgg[i].mult_incr(model.Dgg[i], 1);
    C[i].mult_incr(model.C[i], 1);
    E[i].mult_incr(model.E[i], 1);
  }
}

void ModelHawkesFixedSumExpKernLeastSqList::allocate_weights() {
  C = std::vector<ArrayDouble2d>(n_nodes);
  Dg = std::vector<ArrayDouble>(n_nodes);
  Dgg = std::vector<ArrayDouble2d>(n_nodes);
  E = std::vector<ArrayDouble2d>(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    C[i] = ArrayDouble2d(n_nodes, n_decays);
    C[i].init_to_zero();
    Dg[i] = ArrayDouble(n_decays);
    Dg[i].init_to_zero();
    Dgg[i] = ArrayDouble2d(n_decays, n_decays);
    Dgg[i].init_to_zero();
    E[i] = ArrayDouble2d(n_nodes, n_decays * n_decays);
    E[i].init_to_zero();
  }
  weights_allocated = true;
}

void ModelHawkesFixedSumExpKernLeastSqList::synchronize_aggregated_model() {
  auto *casted_model = static_cast<ModelHawkesFixedSumExpKernLeastSq *>(aggregated_model.get());

  casted_model->n_nodes = n_nodes;
  casted_model->n_decays = n_decays;
  casted_model->max_n_threads = max_n_threads;

  casted_model->C = ArrayDouble2dList1D(n_nodes);
  casted_model->Dg = ArrayDoubleList1D(n_nodes);
  casted_model->Dgg = ArrayDouble2dList1D(n_nodes);
  casted_model->E = ArrayDouble2dList1D(n_nodes);
  // We make views to avoid copies
  for (ulong i = 0; i < n_nodes; ++i) {
    casted_model->Dg[i] = view(Dg[i]);
    casted_model->Dgg[i] = view(Dgg[i]);
    casted_model->C[i] = view(C[i]);
    casted_model->E[i] = view(E[i]);
  }
  casted_model->end_time = end_times->sum();

  casted_model->n_total_jumps = n_jumps_per_realization->sum();
  casted_model->n_jumps_per_node = n_jumps_per_node;

  casted_model->weights_computed = weights_computed;
}

ulong ModelHawkesFixedSumExpKernLeastSqList::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * n_decays;
}
