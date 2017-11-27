// License: BSD 3 clause

//
// Created by Martin Bompaire on 17/06/15.
//

#include "tick/random/test_rand.h"

#include <thread>
#include <chrono>

Rand _init_rand(int seed) {
    return Rand(seed);
}

SArrayIntPtr test_uniform_int(int a,
                              int b,
                              ulong size,
                              int seed) {
    Rand rand(seed);
    SArrayIntPtr sample = SArrayInt::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.uniform_int(a, b);
    }
    return sample;
}

SArrayDoublePtr test_uniform(ulong size,
                             int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.uniform();
    }
    return sample;
}

SArrayDoublePtr test_uniform(double a,
                             double b,
                             ulong size,
                             int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.uniform(a, b);
    }
    return sample;
}

SArrayDoublePtr test_gaussian(ulong size,
                              int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.gaussian();
    }
    return sample;
}

SArrayDoublePtr test_gaussian(double mean,
                              double std,
                              ulong size,
                              int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.gaussian(mean, std);
    }
    return sample;
}

SArrayDoublePtr test_exponential(double intensity,
                                 ulong size,
                                 int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.exponential(intensity);
    }
    return sample;
}

SArrayDoublePtr test_poisson(double rate,
                             ulong size,
                             int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.poisson(rate);
    }
    return sample;
}

SArrayDoublePtr test_discrete(ArrayDouble &probabilities,
                              ulong size,
                              int seed) {
    Rand rand(seed);
    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);

    ulong step1 = size / 4;
    ulong step2 = 2 * size / 4;
    ulong step3 = 3 * size / 4;

    for (ulong i = 0; i < step1; i++) {
        (*sample)[i] = rand.discrete(probabilities);
    }
    rand.set_discrete_dist(probabilities);
    for (ulong i = step1; i < step2; i++) {
        (*sample)[i] = rand.discrete();
    }
    for (ulong i = step2; i < step3; i++) {
        (*sample)[i] = rand.discrete(probabilities);
    }
    for (ulong i = step3; i < size; i++) {
        (*sample)[i] = rand.discrete();
    }
    return sample;
}

SArrayDoublePtr test_uniform_lagged(ulong size,
                                    int wait_time,
                                    int seed) {
    Rand rand(seed);

    SArrayDoublePtr sample = SArrayDouble::new_ptr(size);
    for (ulong i = 0; i < size; i++) {
        (*sample)[i] = rand.uniform();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(wait_time));
    return sample;
}

SArrayDoublePtr test_uniform_threaded(ulong size,
                                      int wait_time,
                                      int seed) {
    SArrayDoublePtr sample;
#ifdef PYTHON_LINK
    Py_BEGIN_ALLOW_THREADS;
#endif
    sample = test_uniform_lagged(size, wait_time, seed);
#ifdef PYTHON_LINK
    Py_END_ALLOW_THREADS;
#endif
    return sample;
}
