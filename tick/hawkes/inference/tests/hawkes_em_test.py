# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes.inference import HawkesEM
from tick.hawkes.model.tests.model_hawkes_test_utils import (
    hawkes_intensities, hawkes_log_likelihood)


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(123269)
        self.n_nodes = 3
        self.n_realizations = 2

        self.events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(self.n_nodes)
        ] for _ in range(self.n_realizations)]

    def test_hawkes_em_attributes(self):
        """...Test attributes of HawkesEM are correctly inherited
        """
        em = HawkesEM(kernel_support=10)
        em.fit(self.events)
        self.assertEqual(em.n_nodes, self.n_nodes)
        self.assertEqual(em.n_realizations, self.n_realizations)

    def test_hawkes_em_fit(self):
        """...Test fit method of HawkesEM
        """
        kernel_support = 3
        kernel_size = 3
        baseline = np.zeros(self.n_nodes) + .2
        kernel = np.zeros((self.n_nodes, self.n_nodes, kernel_size)) + .4

        em = HawkesEM(kernel_support=kernel_support, kernel_size=kernel_size,
                      n_threads=2, max_iter=11, verbose=False)
        em.fit(self.events, baseline_start=baseline, kernel_start=kernel)

        np.testing.assert_array_almost_equal(
            em.baseline, [1.2264, 0.2164, 1.6782], decimal=4)

        expected_kernel = [[[2.4569e-02, 2.5128e-06,
                             0.0000e+00], [1.8072e-02, 5.4332e-11, 0.0000e+00],
                            [2.7286e-03, 4.0941e-08, 3.5705e-15]],
                           [[8.0077e-01, 2.2624e-02,
                             6.7577e-10], [2.7503e-02, 3.1840e-05, 0.0000e+00],
                            [1.4984e-01, 7.8428e-06, 2.8206e-12]],
                           [[1.2163e-01, 1.0997e-02,
                             5.4724e-05], [4.7348e-02, 6.6093e-03, 5.5433e-12],
                            [1.0662e-03, 5.3920e-05, 1.4930e-08]]]

        np.testing.assert_array_almost_equal(em.kernel, expected_kernel,
                                             decimal=4)

        em2 = HawkesEM(
            kernel_discretization=np.array([0., 1., 2., 3.]), n_threads=1,
            max_iter=11, verbose=False)
        em2.fit(self.events, baseline_start=baseline, kernel_start=kernel)
        np.testing.assert_array_almost_equal(em2.kernel, expected_kernel,
                                             decimal=4)

        np.testing.assert_array_almost_equal(
            em.get_kernel_values(1, 0, np.linspace(0, 3, 5)),
            [0.0000e+00, 8.0077e-01, 2.2624e-02, 6.7577e-10, 0.0000e+00],
            decimal=4)

        np.testing.assert_array_almost_equal(
            em.get_kernel_norms(),
            [[0.0246, 0.0181, 0.0027], [0.8234, 0.0275, 0.1499],
             [0.1327, 0.054, 0.0011]], decimal=3)

        np.testing.assert_array_equal(
            em.get_kernel_supports(),
            np.ones((self.n_nodes, self.n_nodes)) * 3)

    def test_hawkes_em_score(self):
        """...Test score (ie. likelihood) function of Hawkes EM
        """

        def approximate_likelihood(em, events, end_times, precision=2):
            n_total_jumps = sum(map(len, events))
            kernels_func = [[
                lambda t, i=i, j=j: em.get_kernel_values(i, j, np.array([t]))[0]
                for j in range(n_nodes)
            ] for i in range(n_nodes)]
            intensities = hawkes_intensities(events, em.baseline, kernels_func)
            return hawkes_log_likelihood(intensities, events, end_times,
                                         precision=precision) / n_total_jumps

        # We use only 2 nodes otherwise integral approximation might be very
        # slow
        n_nodes = 2
        kernel_support = 1
        kernel_size = 3
        baseline = np.random.rand(n_nodes) + .2
        kernel = np.random.rand(n_nodes, n_nodes, kernel_size) + .4

        train_events = \
            [np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]

        test_events = \
            [2 + np.cumsum(np.random.rand(2 + i)) for i in range(n_nodes)]

        # Test for 2 kind of discretization
        train_kwargs = [{
            'kernel_support': 1,
            'kernel_size': 3
        }, {
            'kernel_discretization': np.array([0., 1., 1.5, 3.])
        }]

        # Test with and without fitting
        fits = [True, False]

        for kwargs, fit in zip(train_kwargs, fits):
            em = HawkesEM(**kwargs)
            end_times = max(map(max, train_events)) + 0.2 * kernel_support

            msg = '^You must either call `fit` before `score` or provide events'
            with self.assertRaisesRegex(ValueError, msg):
                em.score()

            if fit:
                em.fit(train_events, end_times=end_times,
                       baseline_start=baseline, kernel_start=kernel)
            else:
                em.baseline = baseline
                em.kernel = kernel

            # Score on em train data
            if fit:
                em_train_score = em.score()
            else:
                em_train_score = em.score(train_events, end_times=end_times)
            self.assertAlmostEqual(
                em_train_score,
                approximate_likelihood(em, train_events, end_times, 2),
                delta=1e-1, msg='Failed on train for {}'.format(kwargs))

            # Score on test data
            em_test_score = em.score(events=test_events)
            test_end_times = max(map(max, test_events))
            self.assertAlmostEqual(
                em_test_score,
                approximate_likelihood(em, test_events, test_end_times, 4),
                delta=1e-3, msg='Failed on test for {}'.format(kwargs))

    def test_hawkes_em_kernel_support(self):
        """...Test that Hawkes em kernel support parameter is correctly
        synchronized
        """
        kernel_support_1 = 4.4
        learner = HawkesEM(kernel_support_1)
        self.assertEqual(learner.kernel_support, kernel_support_1)
        self.assertEqual(learner._learner.get_kernel_support(),
                         kernel_support_1)
        expected_kernel_discretization = [
            0.0, 0.44, 0.88, 1.32, 1.76, 2.2, 2.64, 3.08, 3.52, 3.96, 4.4
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_support_2 = 6.2
        learner.kernel_support = kernel_support_2
        self.assertEqual(learner.kernel_support, kernel_support_2)
        self.assertEqual(learner._learner.get_kernel_support(),
                         kernel_support_2)

        expected_kernel_discretization = [
            0.0, 0.62, 1.24, 1.86, 2.48, 3.1, 3.72, 4.34, 4.96, 5.58, 6.2
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

    def test_hawkes_em_kernel_size(self):
        """...Test that Hawkes em kernel size parameter is correctly
        synchronized
        """
        kernel_size_1 = 4
        learner = HawkesEM(4., kernel_size=kernel_size_1)
        self.assertEqual(learner.kernel_size, kernel_size_1)
        self.assertEqual(learner._learner.get_kernel_size(), kernel_size_1)
        expected_kernel_discretization = [0., 1., 2., 3., 4.]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_size_2 = 5
        learner.kernel_size = kernel_size_2
        self.assertEqual(learner.kernel_size, kernel_size_2)
        self.assertEqual(learner._learner.get_kernel_size(), kernel_size_2)
        expected_kernel_discretization = [0.0, 0.8, 1.6, 2.4, 3.2, 4]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

    def test_hawkes_em_kernel_dt(self):
        """...Test that Hawkes em kernel dt parameter is correctly
        synchronized
        """
        kernel_support = 4
        kernel_size = 10
        learner = HawkesEM(kernel_support, kernel_size=kernel_size)
        self.assertEqual(learner.kernel_dt, 0.4)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), 0.4)

        kernel_dt_1 = 0.2
        learner.kernel_dt = kernel_dt_1
        self.assertEqual(learner.kernel_dt, kernel_dt_1)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(), kernel_dt_1)
        self.assertEqual(learner.kernel_size, 20)
        expected_kernel_discretization = [
            0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6,
            2.8, 3., 3.2, 3.4, 3.6, 3.8, 4.
        ]
        np.testing.assert_array_almost_equal(learner.kernel_discretization,
                                             expected_kernel_discretization)

        kernel_dt_1 = 0.199
        learner.kernel_dt = kernel_dt_1
        self.assertEqual(learner.kernel_dt, 0.19047619047619047)
        self.assertEqual(learner._learner.get_kernel_fixed_dt(),
                         0.19047619047619047)
        self.assertEqual(learner.kernel_size, 21)


if __name__ == "__main__":
    unittest.main()
