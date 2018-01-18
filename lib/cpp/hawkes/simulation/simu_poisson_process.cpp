// License: BSD 3 clause

//
//  poisson.cpp
//  Array
//
//  Created by bacry on 13/04/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//

#include "tick/hawkes/simulation/simu_poisson_process.h"

Poisson::Poisson(double intensity, int seed) : PP(1, seed) {
  intensities = SArrayDouble::new_ptr(1);
  (*intensities)[0] = intensity;
}

Poisson::Poisson(SArrayDoublePtr intensities, int seed) :
  PP(static_cast<unsigned int>(intensities->size()), seed) {
  this->intensities = intensities;
}

void Poisson::init_intensity_(ArrayDouble &intensity, double *total_intensity_bound1) {
  *total_intensity_bound1 = 0;
  for (unsigned int i = 0; i < get_n_nodes(); i++) {
    intensity[i] = (*intensities)[i];
    *total_intensity_bound1 += (*intensities)[i];
  }
}

bool Poisson::update_time_shift_(double delay, ArrayDouble &intensity,
                                 double *total_intensity_bound) {
  return false;
}

