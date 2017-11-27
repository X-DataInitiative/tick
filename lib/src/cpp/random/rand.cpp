// License: BSD 3 clause

//
//  rand.cpp
//  Array
//
//  Created by bacry on 09/04/2015.
//  Copyright (c) 2015 bacry. All rights reserved.
//
//
//  rand.cpp
//  sp
//
//  Created by bacry on 01/03/13.
//
//

#include "tick/random/rand.h"

#include <iostream>
#include <random>

Rand::Rand(int seed)
    : seed(seed) {

    reseed(seed);

    init_reusable_distributions();
}

Rand::Rand(const std::mt19937_64 &generator)
    : seed(0)
    , generator(generator) {
    init_reusable_distributions();
}

void Rand::init_reusable_distributions() {
    // We can easily generate realizations of distributions with other parameters from this ones.
    // Hence we create them once and for all.
    uniform_dist = std::uniform_real_distribution<double>(0, 1);
    normal_dist = std::normal_distribution<double>(0, 1);
}

int Rand::uniform_int(int a, int b) {
    std::uniform_int_distribution<int>::param_type p(a, b);
    return uniform_int_dist(generator, p);
}

ulong Rand::uniform_int(ulong a, ulong b) {
    std::uniform_int_distribution<ulong>::param_type p(a, b);
    return uniform_ulong_dist(generator, p);
}

double Rand::uniform() {
    return uniform_dist(generator);
}

double Rand::uniform(double a, double b) {
    std::uniform_real_distribution<double>::param_type p(a, b);
    return uniform_dist(generator, p);
}

double Rand::gaussian() {
    return normal_dist(generator);
}

double Rand::gaussian(double mean, double std) {
    std::normal_distribution<double>::param_type p(mean, std);
    return normal_dist(generator, p);
}

double Rand::exponential(double intensity) {
    std::exponential_distribution<double>::param_type p(intensity);
    return expon_dist(generator, p);
}

int Rand::poisson(double rate) {
    std::poisson_distribution<int>::param_type p(rate);
    return poisson_dist(generator, p);
}

void Rand::set_discrete_dist(ArrayDouble probabilities) {
    double *start = probabilities.data();
    double *end = probabilities.data() + probabilities.size();
    std::discrete_distribution<ulong>::param_type p(start, end);
    discrete_dist.param(p);
}

ulong Rand::discrete() {
    return discrete_dist(generator);
}

ulong Rand::discrete(ArrayDouble probabilities) {
    double *start = probabilities.data();
    double *end = probabilities.data() + probabilities.size();
    std::discrete_distribution<ulong>::param_type p(start, end);
    return discrete_dist(generator, p);
}

int Rand::get_seed() const {
    return seed;
}

void Rand::reseed(const int seed) {
  // If seed is negative with create one with random device
  // Otherwise we use the given seed
  if (seed < 0) {
    // This random device creates random numbers based on machine state
    std::random_device r;
    // A seed sequence generate random numbers evenly distributed from a given seed
    std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
    generator = std::mt19937_64(seed_seq);
  } else {
    unsigned int useed = seed < 0 ? 0 : static_cast<unsigned int>(seed);
    generator = std::mt19937_64(useed);
  }

  Rand::seed = seed;
}
