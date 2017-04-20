//
// Created by Martin Bompaire on 02/06/15.
//

#define _USE_MATH_DEFINES

#include "hawkes.h"

/// HAWKES

Hawkes::Hawkes(unsigned int dimension1, int seed)
    : PP(dimension1, seed), kernels(n_nodes * n_nodes), mus(n_nodes) {
  for (unsigned int i = 0; i < n_nodes; i++) {
    mus[i] = std::make_shared<HawkesMu>();

    for (unsigned int j = 0; j < n_nodes; j++) {
      kernels[i * n_nodes + j] = std::make_shared<HawkesKernel0>();
    }
  }
}

Hawkes::~Hawkes() {}

void Hawkes::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound) {
  *total_intensity_bound = 0;
  for (unsigned int i = 0; i < n_nodes; i++) {
    intensity[i] = get_mu(i);
    *total_intensity_bound += intensity[i];
  }
}

bool Hawkes::update_time_shift_(double delay,
                                ArrayDouble &intensity,
                                double *total_intensity_bound1) {
  if (total_intensity_bound1) *total_intensity_bound1 = 0;
  bool flag_negative_intensity1 = false;

  // We loop on the contributions
  for (unsigned int i = 0; i < n_nodes; i++) {
    intensity[i] = get_mu(i);
    if (total_intensity_bound1)
      *total_intensity_bound1 += intensity[i];

    for (unsigned int j = 0; j < n_nodes; j++) {
      HawkesKernelPtr &k = kernels[i * n_nodes + j];

      if (k->get_support() == 0) continue;
      double bound = 0;
      intensity[i] += k->get_convolution(get_time() + delay, *timestamps[j], &bound);

      if (total_intensity_bound1) {
        *total_intensity_bound1 += bound;
      }
      if (intensity[i] < 0) flag_negative_intensity1 = true;
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

void Hawkes::set_mu(unsigned int i, const HawkesMuPtr &mu) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  mus[i].reset();

  mus[i] = nullptr;
  if (mu) {
    mus[i] = mu;
  }
}

void Hawkes::set_mu(unsigned int i, double mu) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  set_mu(i, std::make_shared<HawkesMu>(mu));
}

double Hawkes::get_mu(unsigned int i) {
  if (i >= n_nodes) TICK_BAD_INDEX(0, n_nodes, i);

  return mus[i]->get_value();
}


