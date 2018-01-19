// License: BSD 3 clause


#include "tick/hawkes/simulation/simu_hawkes.h"

Hawkes::Hawkes(unsigned int n_nodes, int seed)
  : PP(n_nodes, seed), kernels(n_nodes * n_nodes), baselines(n_nodes) {
  for (unsigned int i = 0; i < n_nodes; i++) {
    baselines[i] = std::make_shared<HawkesConstantBaseline>(0.);

    for (unsigned int j = 0; j < n_nodes; j++) {
      kernels[i * n_nodes + j] = std::make_shared<HawkesKernel0>();
    }
  }
}

void Hawkes::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound) {
  *total_intensity_bound = 0;
  for (unsigned int i = 0; i < n_nodes; i++) {
    intensity[i] = get_baseline(i, 0.);
    *total_intensity_bound += get_baseline_bound(i, 0.);
  }
}

bool Hawkes::update_time_shift_(double delay,
                                ArrayDouble &intensity,
                                double *total_intensity_bound1) {
  if (total_intensity_bound1) *total_intensity_bound1 = 0;
  bool flag_negative_intensity1 = false;

  // We loop on the contributions
  for (unsigned int i = 0; i < n_nodes; i++) {
    intensity[i] = get_baseline(i, get_time());
    if (total_intensity_bound1)
      *total_intensity_bound1 += get_baseline_bound(i, get_time());

    for (unsigned int j = 0; j < n_nodes; j++) {
      HawkesKernelPtr &k = kernels[i * n_nodes + j];

      if (k->get_support() == 0) continue;
      double bound = 0;
      intensity[i] += k->get_convolution(get_time() + delay, *timestamps[j], &bound);

      if (total_intensity_bound1) {
        *total_intensity_bound1 += bound;
      }
      if (intensity[i] < 0) {
        if (threshold_negative_intensity) intensity[i] = 0;
        flag_negative_intensity1 = true;
      }
    }
  }
  return flag_negative_intensity1;
}

void Hawkes::reset() {
  for (unsigned int i = 0; i < n_nodes; i++) {
    for (unsigned int j = 0; j < n_nodes; j++) {
      if (kernels[i * n_nodes + j] != nullptr)
        kernels[i * n_nodes + j]->rewind();
    }
  }
  PP::reset();
}

void Hawkes::set_kernel(unsigned int i, unsigned int j, HawkesKernelPtr &kernel) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);
  if (j >= n_nodes) TICK_BAD_INDEX(0, n_nodes, j);

  kernels[i * n_nodes + j].reset();

  if (kernel == nullptr)
    kernel = std::make_shared<HawkesKernel>();
  else
    kernel = kernel->duplicate_if_necessary(kernel);
  kernels[i * n_nodes + j] = kernel;
}

HawkesKernelPtr Hawkes::get_kernel(unsigned int i, unsigned int j) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);
  if (j >= n_nodes) TICK_BAD_INDEX(0, n_nodes, j);

  return kernels[i * n_nodes + j];
}

void Hawkes::set_baseline(unsigned int i, const HawkesBaselinePtr &baseline) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  if (baseline) {
    baselines[i] = baseline;
  }
}

void Hawkes::set_baseline(unsigned int i, double baseline) {
  set_baseline(i, std::make_shared<HawkesConstantBaseline>(baseline));
}

void Hawkes::set_baseline(unsigned int i, TimeFunction time_function) {
  set_baseline(i, std::make_shared<HawkesTimeFunctionBaseline>(time_function));
}

void Hawkes::set_baseline(unsigned int i, ArrayDouble &times, ArrayDouble &values) {
  set_baseline(i, std::make_shared<HawkesTimeFunctionBaseline>(times, values));
}

double Hawkes::get_baseline(unsigned int i, double t) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  return baselines[i]->get_value(t);
}

SArrayDoublePtr Hawkes::get_baseline(unsigned int i, ArrayDouble &t) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  return baselines[i]->get_value(t);
}

double Hawkes::get_baseline_bound(unsigned int i, double t) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  return baselines[i]->get_future_bound(t);
}


