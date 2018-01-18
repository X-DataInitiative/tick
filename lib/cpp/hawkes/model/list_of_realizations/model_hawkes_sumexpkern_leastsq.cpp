// License: BSD 3 clause


#include "tick/hawkes/model/list_of_realizations/model_hawkes_sumexpkern_leastsq.h"

ModelHawkesSumExpKernLeastSq::ModelHawkesSumExpKernLeastSq(
  const ArrayDouble &decays,
  const ulong n_baselines,
  const double period_length,
  const unsigned int max_n_threads,
  const unsigned int optimization_level)
  : ModelHawkesLeastSq(max_n_threads, optimization_level),
    n_baselines(n_baselines), period_length(period_length),
    decays(decays), n_decays(decays.size()) {
  aggregated_model = std::unique_ptr<ModelHawkesSumExpKernLeastSqSingle>(
    new ModelHawkesSumExpKernLeastSqSingle(decays, n_baselines, period_length,
                                           max_n_threads, optimization_level));
}

void ModelHawkesSumExpKernLeastSq::compute_weights_i_r(
  const ulong i_r, std::vector<ModelHawkesSumExpKernLeastSqSingle> &model_list) {
  const ulong r = static_cast<const ulong>(i_r / n_nodes);
  const ulong i = i_r % n_nodes;

  model_list[r].compute_weights_i(i);
}

void ModelHawkesSumExpKernLeastSq::compute_weights_timestamps_list() {
  auto model_list =
    std::vector<ModelHawkesSumExpKernLeastSqSingle>(n_realizations);

  for (ulong r = 0; r < n_realizations; ++r) {
    model_list[r] = ModelHawkesSumExpKernLeastSqSingle(decays, n_baselines, period_length,
                                                       1, optimization_level);
    model_list[r].set_data(timestamps_list[r], (*end_times)[r]);
    model_list[r].allocate_weights();
  }

  // Multithreaded computation of the arrays
  parallel_run(get_n_threads(), n_realizations * n_nodes,
               &ModelHawkesSumExpKernLeastSq::compute_weights_i_r, this, model_list);

  for (ulong r = 0; r < n_realizations; ++r) {
    L.mult_incr(model_list[r].L, 1);
    for (ulong i = 0; i < n_nodes; ++i) {
      Dg[i].mult_incr(model_list[r].Dg[i], 1);
      Dgg[i].mult_incr(model_list[r].Dgg[i], 1);
      C[i].mult_incr(model_list[r].C[i], 1);
      E[i].mult_incr(model_list[r].E[i], 1);
      K[i].mult_incr(model_list[r].K[i], 1);
    }
  }
}

void ModelHawkesSumExpKernLeastSq::compute_weights_timestamps(
  const SArrayDoublePtrList1D &timestamps, double end_time) {
  auto model = ModelHawkesSumExpKernLeastSqSingle(decays, n_baselines, period_length,
                                                  get_n_threads(), optimization_level);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  L.mult_incr(model.L, 1);
  for (ulong i = 0; i < n_nodes; ++i) {
    Dg[i].mult_incr(model.Dg[i], 1);
    Dgg[i].mult_incr(model.Dgg[i], 1);
    C[i].mult_incr(model.C[i], 1);
    E[i].mult_incr(model.E[i], 1);
    K[i].mult_incr(model.K[i], 1);
  }
}

void ModelHawkesSumExpKernLeastSq::allocate_weights() {
  L = ArrayDouble(n_baselines);
  L.init_to_zero();

  C = std::vector<ArrayDouble2d>(n_nodes);
  Dgg = std::vector<ArrayDouble2d>(n_nodes);
  E = std::vector<ArrayDouble2d>(n_nodes);
  Dg = ArrayDouble2dList1D(n_nodes);
  K = ArrayDoubleList1D(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    C[i] = ArrayDouble2d(n_nodes, n_decays);
    C[i].init_to_zero();
    Dg[i] = ArrayDouble2d(n_decays, n_baselines);
    Dg[i].init_to_zero();
    Dgg[i] = ArrayDouble2d(n_decays, n_decays);
    Dgg[i].init_to_zero();
    E[i] = ArrayDouble2d(n_nodes, n_decays * n_decays);
    E[i].init_to_zero();
    K[i] = ArrayDouble(n_baselines);
    K[i].init_to_zero();
  }
  weights_allocated = true;
}

void ModelHawkesSumExpKernLeastSq::synchronize_aggregated_model() {
  auto *casted_model = static_cast<ModelHawkesSumExpKernLeastSqSingle *>(aggregated_model.get());

  casted_model->n_nodes = n_nodes;
  casted_model->n_decays = n_decays;
  casted_model->n_baselines = n_baselines;
  casted_model->period_length = period_length;
  casted_model->max_n_threads = max_n_threads;

  casted_model->L = view(L);
  casted_model->C = ArrayDouble2dList1D(n_nodes);
  casted_model->Dg = ArrayDouble2dList1D(n_nodes);
  casted_model->Dgg = ArrayDouble2dList1D(n_nodes);
  casted_model->E = ArrayDouble2dList1D(n_nodes);
  casted_model->K = ArrayDoubleList1D(n_nodes);
  // We make views to avoid copies
  for (ulong i = 0; i < n_nodes; ++i) {
    casted_model->Dg[i] = view(Dg[i]);
    casted_model->Dgg[i] = view(Dgg[i]);
    casted_model->C[i] = view(C[i]);
    casted_model->E[i] = view(E[i]);
    casted_model->K[i] = view(K[i]);
  }
  casted_model->end_time = end_times->sum();

  casted_model->n_total_jumps = n_jumps_per_realization->sum();
  casted_model->n_jumps_per_node = n_jumps_per_node;

  casted_model->weights_computed = weights_computed;
}

ulong ModelHawkesSumExpKernLeastSq::get_n_coeffs() const {
  return n_nodes * n_baselines + n_nodes * n_nodes * n_decays;
}

ulong ModelHawkesSumExpKernLeastSq::get_n_baselines() const {
  return n_baselines;
}

void ModelHawkesSumExpKernLeastSq::set_n_baselines(ulong n_baselines) {
  this->n_baselines = n_baselines;
  weights_computed = false;
}

double ModelHawkesSumExpKernLeastSq::get_period_length() const {
  return period_length;
}

void ModelHawkesSumExpKernLeastSq::set_period_length(double period_length) {
  this->period_length = period_length;
  weights_computed = false;
}
