// License: BSD 3 clause

//
// Created by Martin Bompaire on 24/11/15.
//

#include "tick/hawkes/simulation/simu_inhomogeneous_poisson.h"

InhomogeneousPoisson::InhomogeneousPoisson(const TimeFunction &intensities_function,
                                           int seed)
  : PP(1, seed), intensities_functions(1) {
  intensities_functions[0] = intensities_function;
}

InhomogeneousPoisson::InhomogeneousPoisson(const std::vector<TimeFunction> &intensities_functions,
                                           int seed)
  : PP(static_cast<unsigned int>(intensities_functions.size()), seed),
    intensities_functions(intensities_functions) {
}

void InhomogeneousPoisson::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound1) {
  *total_intensity_bound1 = 0;

  for (unsigned int i = 0; i < get_n_nodes(); i++) {
    intensity[i] = intensities_functions[i].value(get_time());
    intensities_functions[i].compute_future_max();
    *total_intensity_bound1 += intensities_functions[i].future_bound(get_time());
  }
}

bool InhomogeneousPoisson::update_time_shift_(double delay,
                                              ArrayDouble &intensity,
                                              double *total_intensity_bound1) {
  if (total_intensity_bound1) *total_intensity_bound1 = 0;

  bool flag_negative_intensity1 = false;

  for (unsigned int i = 0; i < get_n_nodes(); i++) {
    intensity[i] = intensities_functions[i].value(get_time() + delay);
    if (total_intensity_bound1)
      *total_intensity_bound1 += intensities_functions[i].future_bound(get_time() + delay);
    if (intensity[i] < 0) flag_negative_intensity1 = true;
  }

  return flag_negative_intensity1;
}

