// License: BSD 3 clause

//
// Created by Martin Bompaire on 02/03/2017.
//

#include "tick/hawkes/model/base/model_hawkes_single.h"

ModelHawkesSingle::ModelHawkesSingle(const int max_n_threads,
                                     const unsigned int optimization_level)
    : ModelHawkes(max_n_threads, optimization_level), n_total_jumps(0) {}

void ModelHawkesSingle::set_data(const SArrayDoublePtrList1D &timestamps,
                                 const double end_time) {
  weights_computed = false;

  n_nodes = timestamps.size();
  set_n_nodes(n_nodes);

  n_total_jumps = 0;
  n_jumps_per_node = SArrayULong::new_ptr(n_nodes);
  for (ulong i = 0; i < n_nodes; ++i) {
    (*n_jumps_per_node)[i] = timestamps[i]->size();
  }
  n_total_jumps = n_jumps_per_node->sum();

  for (ulong i = 0; i < n_nodes; ++i) {
    if (timestamps[i]->size() > 0) {
      double last_time_i = (*timestamps[i])[timestamps[i]->size() - 1];
      if (end_time < last_time_i) {
        TICK_ERROR("Provided end_time ("
                   << end_time << ") is smaller than last time of "
                   << "component " << i << " (" << last_time_i << ")")
      }
    }
  }

  this->end_time = end_time;
  this->timestamps = timestamps;
}

unsigned int ModelHawkesSingle::get_n_threads() const {
  return std::min(this->max_n_threads, static_cast<unsigned int>(n_nodes));
}


double ModelHawkesSingle::compute_bjk(unsigned long k, double betajk) {
  auto kernel = [betajk](double t) { return betajk * exp(-betajk * t); };
  double hjk = betajk; // This will be Hjk(t^k_n)
  double bjk = -1;

  for (unsigned long n = 1; n < timestamps[k]->size(); n++) {
    hjk = kernel((*timestamps[k])[n] - (*timestamps[k])[n - 1]) * hjk / betajk + betajk;
    if (bjk < hjk) bjk = hjk;
  }
  return bjk;
}

double ModelHawkesSingle::compute_vjk(unsigned long j,
                                                    unsigned long k,
                                                    double betajk) {
  auto kernel = [betajk](double t) { return betajk * exp(-betajk * t); };
  double hjk_ = 0; // This will be Hjk(t^j_n)

  // We first compute it for n= 0
  unsigned long n = 0;
  unsigned long p;
  for (p = 0; p < timestamps[k]->size() && (*timestamps[k])[p] <= (*timestamps[j])[n]; p++) {
    hjk_ += kernel((*timestamps[j])[n] - (*timestamps[k])[p]);
  }
  double vjk = hjk_ * hjk_;
  // Now we loop on n >= 1
  n++;
  for (; n < timestamps[j]->size(); n++) {
    hjk_ *= kernel((*timestamps[j])[n] - (*timestamps[j])[n - 1]) / betajk;
    for (; p < timestamps[k]->size() && (*timestamps[k])[p] <= (*timestamps[j])[n]; p++) {
      hjk_ += kernel((*timestamps[j])[n] - (*timestamps[k])[p]);
    }
    vjk += hjk_ * hjk_;
  }
  vjk /= end_time;
  return vjk;
}

