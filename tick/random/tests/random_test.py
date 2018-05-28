# License: BSD 3 clause

# -*- coding: utf8 -*-
import unittest
import multiprocessing
from multiprocessing.pool import Pool

import itertools
import numpy as np
import threading
from scipy import stats
from tick.random import test_uniform, test_gaussian, test_poisson, \
    test_exponential, test_uniform_int, test_discrete, test_uniform_threaded


class Test(unittest.TestCase):
    def setUp(self):
        self.test_size = 5
        self.test_seed = 12098
        self.stat_size = 10000
        self.thread_types = ['multiprocessing', 'threading']

    def assert_samples_are_different(self, sample1, samples2, discrete):
        # if law is continuous
        if not discrete:
            self.assertGreater(np.min(np.abs(sample1 - samples2)), 1e-7)
        else:
            self.assertGreater(np.max(np.abs(sample1 - samples2)), 0)

    def _test_dist_with_seed(self, seeded_sample, test_function, *args,
                             discrete=False):

        # arguments given to test function, we append size and seed
        # to other arguments
        seed_args = list(args) + [self.test_size, self.test_seed]
        seeded_sample_1 = test_function(*seed_args)
        seeded_sample_2 = test_function(*seed_args)

        # We check that samples we same seed are equal
        np.testing.assert_almost_equal(seeded_sample_1, seeded_sample_2)

        # This is temporary
        # At the moment, seeds are not cross platform as distributions in
        # C++ depends on standard library shipped with compiler
        import os
        if os.name == 'posix':
            import platform
            if platform.system() == 'Darwin':
                # We check that we get the same as what was recorded
                np.testing.assert_almost_equal(seeded_sample_1, seeded_sample)

        # arguments given to test function, we append size and seed to other
        # arguments
        other_seed_args = list(args) + [self.test_size, self.test_seed + 1]
        sample_other_seed = test_function(*other_seed_args)
        self.assert_samples_are_different(seeded_sample, sample_other_seed,
                                          discrete)

        # arguments given to test function, we append size to other arguments
        no_seed_args = list(args) + [self.test_size]
        sample_no_seed = test_function(*no_seed_args)
        self.assert_samples_are_different(seeded_sample, sample_no_seed,
                                          discrete)

    def test_uniform_int_random(self):
        """...Test uniform random int numbers in range simulation
        """
        a = -2
        b = 100
        seeded_sample = \
            [28, 30, 54, 74, 11]

        self._test_dist_with_seed(seeded_sample, test_uniform_int, a, b,
                                  discrete=True)

        # Statistical tests
        sample = test_uniform_int(a, b, self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'randint', (a, b))
        self.assertLess(p, 0.05)

    def test_uniform_random(self):
        """...Test uniform random numbers simulation
        """
        seeded_sample = \
            [0.21519787, 0.34657955, 0.52366921, 0.0583405, 0.83635939]

        self._test_dist_with_seed(seeded_sample, test_uniform)

        # Statistical tests
        sample = test_uniform(self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'uniform')
        self.assertLess(p, 0.05)

    def test_uniform_random_with_bounds(self):
        """...Test uniform random numbers with bounds simulation
        """
        a = -2
        b = 5

        seeded_sample = \
            [-0.4936149, 0.42605685, 1.66568447, -1.59161653, 3.85451574]

        self._test_dist_with_seed(seeded_sample, test_uniform, a, b)

        # Statistical tests
        sample = test_uniform(a, b, self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'uniform', (a, b - a))
        self.assertLess(p, 0.05)

    def test_gaussian_random(self):
        """...Test gaussian random numbers simulation
        """
        seeded_sample = \
            [-1.1618693, -0.62588897, 0.03748094, -0.69938171, 0.35105305]

        self._test_dist_with_seed(seeded_sample, test_gaussian)

        # Statistical tests
        sample = test_gaussian(self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'norm')
        self.assertLess(p, 0.05)

    def test_gaussian_random_with_bounds(self):
        """...Test gaussian random numbers simulation with mean and scale
        defined
        """
        mu = -10
        sigma = 0.5

        seeded_sample = \
            [-10.58093465, -10.31294449, -9.98125953, -10.34969085, -9.82447348]

        self._test_dist_with_seed(seeded_sample, test_gaussian, mu, sigma)

        # Statistical tests
        sample = test_gaussian(mu, sigma, self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'norm', (mu, sigma))
        self.assertLess(p, 0.05)

    def test_exponential_random(self):
        """...Test exponential random numbers simulation
        """
        intensity = 1.576

        seeded_sample = \
            [0.1537587, 0.2700092, 0.4705855, 0.0381418, 1.1485296]

        self._test_dist_with_seed(seeded_sample, test_exponential, intensity)

        # Statistical tests
        sample = test_exponential(intensity, self.stat_size, self.test_seed)
        p, _ = stats.kstest(sample, 'expon', (0, 1. / intensity))
        self.assertLess(p, 0.05)

    def test_poisson_random(self):
        """...Test Poisson random numbers simulation
        """
        rate = 5
        seeded_sample = [3., 6., 2., 4., 3.]

        self._test_dist_with_seed(seeded_sample, test_poisson, rate,
                                  discrete=True)

        # Statistical tests
        # We take a smaller sample as chi2 test is expensive
        sample = test_poisson(rate, int(self.stat_size * 0.05))

        # To test statistical consistency of poisson we do like if it was a
        # discrete law with a probability of sum_{k>K}(P(k)) for the last event
        probabilities = [
            stats.poisson.pmf(i, rate) for i in range(3 * int(rate))
        ]
        f_obs = [sum(sample == i) for i in range(len(probabilities))]

        # We add the last event
        f_obs.append(sum(sample >= len(probabilities)))
        probabilities.append(1 - sum(probabilities))

        _, p = stats.chisquare(f_exp=probabilities, f_obs=f_obs)
        self.assertLess(p, 0.05)

    def test_discrete_random(self):
        """...Test discrete random numbers simulation
        """
        probabilities = np.array([2.0, 0.1, 3, 5, 7])
        seeded_sample = [2., 3., 3., 0., 4.]

        self._test_dist_with_seed(seeded_sample, test_discrete, probabilities,
                                  discrete=True)

        # Statistical tests
        sample = test_discrete(probabilities, self.stat_size, self.test_seed)
        f_obs = [sum(sample == i) for i in range(len(probabilities))]
        _, p = stats.chisquare(f_exp=probabilities, f_obs=f_obs)
        self.assertLess(p, 0.05)

        # Test that variable event with probability 0 never happens
        probabilities_zero = probabilities.copy()
        probabilities_zero[1] = 0
        sample = test_discrete(probabilities_zero, self.stat_size,
                               self.test_seed)
        self.assertEqual(sum(sample == 1), 0)

    def _generate_samples_in_parallel(self, n_task=10, n_workers=None,
                                      wait_time=0,
                                      parallelization_type='multiprocessing'):
        """This function generates samples of uniform distribution in parallel

        Parameters
        ----------
        n_task : `int`
            Number of task that will be run

        n_workers : `int`
            Number of workers that will be used.
            If it is set to None the maximum number of workers will be used.
            Otherwise this number of processes will be created (in case of
            multiprocessing) or work will be executed sequestially (in case
            of threading)

        wait_time : `float`
            Time that will be waited on each thread after random number
            simulation

        parallelization_type : `string`
            How work will be parallelized : either using multiprocessing or
            using threading
        """
        sample_size = self.stat_size
        args = [(sample_size, wait_time) for _ in range(n_task)]

        if parallelization_type == 'multiprocessing':

            with Pool(processes=n_workers) as pool:
                samples = pool.starmap(test_uniform_threaded, args)

            samples = np.array(samples)

        elif parallelization_type == 'threading':

            class UniformThread(threading.Thread):
                def __init__(self, sample_size, wait_time):
                    super(UniformThread, self).__init__()
                    self.sample_size = sample_size
                    self.wait_time = wait_time
                    self.sample = None

                def run(self):
                    self.sample = test_uniform_threaded(
                        self.sample_size, self.wait_time)

            if n_workers is None:
                threads = []
                for arg in args:
                    threads.append(UniformThread(*arg))

                [t.start() for t in threads]
                [t.join() for t in threads]
                samples = np.array([t.sample for t in threads])

            else:
                samples = np.array(
                    [test_uniform_threaded(*arg) for arg in args])
        else:
            raise ValueError('Unknown thread type')

        return samples

    def test_parallel_create_independant_random(self):
        """...Test that random number generator creates independant
        samples in a multithreaded environment
        """

        for thread_type in self.thread_types:
            samples = self._generate_samples_in_parallel(
                parallelization_type=thread_type)

            # We check that we do not have any lines that is identical to the
            # one following
            following_samples_are_different = \
                np.prod(np.linalg.norm(samples[:-1] - samples[1:], axis=1) > 0)
            self.assertEqual(
                following_samples_are_different, 1,
                "Two samples generated in parallel are identical")

            # We check that our generated samples are not correlated
            for (i, sample_i), (j, sample_j) in \
                    itertools.product(enumerate(samples), enumerate(samples)):
                if i != j:
                    corr_coeff = stats.pearsonr(sample_i, sample_j)[0]
                    self.assertLess(np.abs(corr_coeff), 0.1)

    def test_parallel_statistical_significance(self):
        """...Test that samples generated in parallel are statistically coherent
        """
        n_task = 10
        for thread_type in self.thread_types:
            samples = self._generate_samples_in_parallel(
                parallelization_type=thread_type, n_task=10)

            for sample in samples:
                # compute p-value with Kolmogorov–Smirnov test
                p, _ = stats.kstest(sample, 'uniform')
                self.assertLess(p, 0.05)

            samples.resize((n_task * self.stat_size,))

            # compute p-value with Kolmogorov–Smirnov test
            p, _ = stats.kstest(samples, 'uniform')
            self.assertLess(p, 0.05)

    """
    def test_parallel_speed_improvement(self):
        \"""...Test our samples are really generated in parallel
        \"""
        wait_time = 10000
        if multiprocessing.cpu_count() > 1:
            for thread_type in self.thread_types:
                start = time.perf_counter()
                self._generate_samples_in_parallel(
                    parallelization_type=thread_type, wait_time=wait_time,
                    n_workers=1)
                time_needed_sequential = time.perf_counter() - start
                start = time.perf_counter()
                
                # With multiprocessing it is counterproductive to have to many
                # process for too small tasks
                if thread_type == 'multiprocessing':
                    n_workers = min(4, multiprocessing.cpu_count())
                else:
                    n_workers = None
                
                self._generate_samples_in_parallel(
                    parallelization_type=thread_type, wait_time=wait_time,
                    n_workers=n_workers)
                time_needed_parallel = time.perf_counter() - start

                # check there is at least a small speed up
                self.assertGreater(
                    time_needed_sequential / time_needed_parallel, 1.5)
    """


if __name__ == "__main__":
    unittest.main()
