//
// Created by Martin Bompaire on 17/06/15.
//

#ifndef LIB_INCLUDE_TICK_RANDOM_TEST_RAND_H_
#define LIB_INCLUDE_TICK_RANDOM_TEST_RAND_H_

// License: BSD 3 clause

#include "tick/base/defs.h"
#include "tick/array/sarray.h"
#include "rand.h"

/**
 * @brief Test simulation of uniform random int numbers in range
 * \param a : lower bound of the range
 * \param b : upper bound of the range
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 */
SArrayIntPtr test_uniform_int(int a, int b, ulong size, int seed = -1);

/**
 * @brief Test simulation of uniform random numbers between 0 and 1
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 */
SArrayDoublePtr test_uniform(ulong size, int seed = -1);

/**
 * @brief Test simulation of uniform random numbers in a range
 * \param a : lower bound of the range
 * \param b : upper bound of the range
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 */
SArrayDoublePtr test_uniform(double a, double b, ulong size, int seed = -1);

/**
 * @brief Test simulation of gaussian random numbers
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 */
SArrayDoublePtr test_gaussian(ulong size, int seed = -1);

/**
 * @brief Test simulation of gaussian random numbers with given mean and standard deviation
 * \param mean : mean of the gaussian distribution
 * \param std : standard deviation of the gaussian distribution
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 */
SArrayDoublePtr test_gaussian(double mean, double std, ulong size, int seed = -1);

/**
 * @brief Test simulation of random numbers following exponential distribution with given intensity
 * \param intensity : intensity of the exponential distribution
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 */
SArrayDoublePtr test_exponential(double intensity, ulong size, int seed = -1);

/**
 * @brief Test simulation of random numbers following Poisson distribution with given rate
 * \param rate : rate of the Poisson distribution
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 */
SArrayDoublePtr test_poisson(double rate, ulong size, int seed = -1);

/**
 * @brief Test simulation of random numbers following a discrete distribution with given
 * probabilities for each event
 * \param probabilities : probabilities of each event
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \returns : The generated sample
 * \note : this test runs both sample generation with probabilities set in advance and with
 * probabilities given on the fly (2 different methods of the random class)
 */
SArrayDoublePtr test_discrete(ArrayDouble &probabilities, ulong size, int seed = -1);

/**
 * @brief Test simulation of uniform random numbers between 0 and 1 with a wait time
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \param wait_time : time waited after simulation
 * \returns : The generated sample
 * \note : We add a wait time as this function will be used in order to see if our code is run in
 * parallel. As sleep is supposed to have a speedup close to 1 we would be able to check if we gain
 * time.
 */
SArrayDoublePtr test_uniform_lagged(ulong size, int wait_time = 0, int seed = -1);

/**
 * @brief Test simulation of uniform random numbers between 0 and 1 with a wait time
 * \param size : size of the simulated sample
 * \param seed : seed of the random generator. If negative, a random seed will be taken
 * \param wait_time : time waited after simulation
 * \returns : The generated sample
 * \note : This function launch the previous one but is allowed to be run on several threads
 * simultaneously
 */
SArrayDoublePtr test_uniform_threaded(ulong size, int wait_time = 0, int seed = -1);

#endif  // LIB_INCLUDE_TICK_RANDOM_TEST_RAND_H_
