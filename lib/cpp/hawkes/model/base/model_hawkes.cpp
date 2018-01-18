// License: BSD 3 clause


#include "tick/hawkes/model/base/model_hawkes.h"

ModelHawkes::ModelHawkes(const int max_n_threads,
                         const unsigned int optimization_level) :
  optimization_level(optimization_level),
  weights_computed(false), n_nodes(0) {
  set_n_threads(max_n_threads);
  n_jumps_per_node = SArrayULong::new_ptr(n_nodes);
}

void ModelHawkes::set_n_nodes(const ulong n_nodes) {
  if (n_nodes == 0) {
    TICK_ERROR("Your realization should have more than one node");
  }
  this->n_nodes = n_nodes;
}

void ModelHawkes::set_n_threads(const int max_n_threads) {
  this->max_n_threads = max_n_threads >= 1 ? static_cast<unsigned int>(max_n_threads)
                                           : std::thread::hardware_concurrency();
}
