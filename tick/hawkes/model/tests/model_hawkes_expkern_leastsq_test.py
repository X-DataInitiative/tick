# License: BSD 3 clause

import pickle
import unittest

import numpy as np
from scipy.optimize import check_grad

from tick.hawkes.model import ModelHawkesExpKernLeastSq
from tick.hawkes.model.tests.model_hawkes_test_utils import (
    hawkes_exp_kernel_intensities, hawkes_least_square_error)


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(30732)

        self.n_nodes = 3
        self.n_realizations = 2

        self.decays = np.random.rand(self.n_nodes, self.n_nodes)

        self.timestamps_list = [[
            np.cumsum(np.random.random(np.random.randint(3, 7)))
            for _ in range(self.n_nodes)
        ] for _ in range(self.n_realizations)]

        self.baseline = np.random.rand(self.n_nodes)
        self.adjacency = np.random.rand(self.n_nodes, self.n_nodes)
        self.coeffs = np.hstack((self.baseline, self.adjacency.ravel()))

        self.realization = 0
        self.model = \
            ModelHawkesExpKernLeastSq(decays=self.decays)
        self.model.fit(self.timestamps_list[self.realization])

        self.model_list = \
            ModelHawkesExpKernLeastSq(decays=self.decays)
        self.model_list.fit(self.timestamps_list)

    def test_model_hawkes_losses(self):
        """...Test that computed losses are consistent with approximated
        theoretical values
        """
        timestamps = self.timestamps_list[self.realization]

        intensities = hawkes_exp_kernel_intensities(self.baseline, self.decays,
                                                    self.adjacency, timestamps)

        integral_approx = hawkes_least_square_error(
            intensities, timestamps, self.model.end_times[self.realization])
        integral_approx /= self.model.n_jumps

        self.assertAlmostEqual(integral_approx, self.model.loss(self.coeffs),
                               places=2)

    def test_model_hawkes_least_sq_multiple_events(self):
        """...Test that multiple events list for ModelHawkesExpKernLeastSq
        is consistent with direct integral estimation
        """
        end_times = np.array([max(map(max, e)) for e in self.timestamps_list])
        end_times += 1.
        self.model_list.fit(self.timestamps_list, end_times=end_times)

        intensities_list = [
            hawkes_exp_kernel_intensities(self.baseline, self.decays,
                                          self.adjacency, timestamps)
            for timestamps in self.timestamps_list
        ]

        integral_approx = sum([
            hawkes_least_square_error(intensities, timestamps, end_time)
            for (intensities, timestamps,
                 end_time) in zip(intensities_list, self.timestamps_list,
                                  self.model_list.end_times)
        ])

        integral_approx /= self.model_list.n_jumps
        self.assertAlmostEqual(integral_approx,
                               self.model_list.loss(self.coeffs), places=2)

    def test_model_hawkes_least_sq_incremental_fit(self):
        """...Test that multiple events list for ModelHawkesExpKernLeastSq
        are correctly handle with incremental_fit
        """
        model_incremental_fit = \
            ModelHawkesExpKernLeastSq(decays=self.decays)

        for timestamps in self.timestamps_list:
            model_incremental_fit.incremental_fit(timestamps)

        self.assertEqual(
            model_incremental_fit.loss(self.coeffs),
            self.model_list.loss(self.coeffs))

    def test_model_hawkes_least_sq_grad(self):
        """...Test that ModelHawkesExpKernLeastSq gradient is consistent
        with loss
        """

        for model in [self.model, self.model_list]:
            self.assertLess(
                check_grad(model.loss, model.grad, self.coeffs), 1e-5)

    def test_ModelHawkesExpKernLeastSqHess(self):
        """...Numerical consistency check of hessian for Hawkes contrast
        """
        for model in [self.model, self.model_list]:
            # this hessian is independent of x but for more generality
            # we still put an used coeff as argument
            hessian = model.hessian(self.coeffs).todense()

            # Check that hessian is equal to its transpose
            np.testing.assert_array_almost_equal(hessian, hessian.T,
                                                 decimal=10)

            # Check that for all dimension hessian row is consistent
            # with its corresponding gradient coordinate.
            for i in range(model.n_coeffs):

                def g_i(x):
                    return model.grad(x)[i]

                def h_i(x):
                    return np.asarray(hessian)[i, :]

                self.assertLess(check_grad(g_i, h_i, self.coeffs), 1e-5)

    def test_model_hawkes_least_sq_change_decays(self):
        """...Test that loss is still consistent after decays modification in
        ModelHawkesExpKernLeastSq
        """
        decays = np.random.rand(self.n_nodes, self.n_nodes)

        self.assertNotEqual(decays[0, 0], self.decays[0, 0])

        model_change_decay = ModelHawkesExpKernLeastSq(decays=decays)
        model_change_decay.fit(self.timestamps_list)
        loss_old_decay = model_change_decay.loss(self.coeffs)

        model_change_decay.decays = self.decays

        self.assertNotEqual(loss_old_decay,
                            model_change_decay.loss(self.coeffs))

        self.assertEqual(
            self.model_list.loss(self.coeffs),
            model_change_decay.loss(self.coeffs))

    def test_hawkes_list_n_threads(self):
        """...Test that the number of used threads is as expected
        """
        model_contrast_list = \
            ModelHawkesExpKernLeastSq(decays=self.decays, n_threads=1)

        # 0 threads yet as no data has been given
        self.assertEqual(model_contrast_list._model.get_n_threads(), 0)

        # Now that it has been fitted it equals
        # min(n_threads, n_nodes * n_realizations)
        model_contrast_list.fit(self.timestamps_list)
        self.assertEqual(model_contrast_list._model.get_n_threads(), 1)

        model_contrast_list.n_threads = 8
        self.assertEqual(model_contrast_list._model.get_n_threads(), 6)

        realization_2_nodes = [np.array([3., 4.]), np.array([3.5, 6.])]

        model_contrast_list.fit(realization_2_nodes)
        self.assertEqual(model_contrast_list._model.get_n_threads(), 2)

        model_contrast_list.n_threads = 1
        self.assertEqual(model_contrast_list._model.get_n_threads(), 1)

    # deprecated test
    def test_ModelHawkesExpKernLeastSqApprox0(self):
        """...Numerical consistency check of lik and grad for Hawkes
        Least-Squares with approx=0
        """
        timestamps = [
            np.array([.2, .3, .65, .87, 1, 10, 12, 22]),
            np.array([3., 40., 60.])
        ]
        beta = 2.

        model = ModelHawkesExpKernLeastSq(decays=beta).fit(timestamps)

        coeffs = np.array([.1, .4, .3, 1., .4, .5])
        grad = np.zeros(6)

        loss = model._loss(coeffs)
        self.assertAlmostEqual(loss, 1.05752053, delta=1e-7)

        model._grad(coeffs, grad)
        test = np.array([
            0.4363636, 4.5818182, -0.6009268, 0.4027132, 1.8310919, 0.3308908
        ])
        np.testing.assert_almost_equal(grad, test, decimal=7)

        loss = model._loss_and_grad(coeffs, grad)
        np.testing.assert_almost_equal(grad, test, decimal=7)
        self.assertAlmostEqual(loss, 1.05752053, delta=1e-7)

    def test_ModelHawkesExpKernLeastSqApprox1(self):
        """...Numerical consistency check of lik and grad for Hawkes
        Least-Squares with approx=1
        """
        timestamps = [
            np.array([.2, .3, .65, .87, 1, 10, 12, 22]),
            np.array([3., 40., 60.])
        ]
        beta = 2.

        model = ModelHawkesExpKernLeastSq(decays=beta,
                                          approx=1).fit(timestamps)

        coeffs = np.array([.1, .4, .3, 1., .4, .5])
        grad = np.zeros(6)

        loss = model._loss(coeffs)
        self.assertAlmostEqual(loss, 1.05752053, delta=1e-4)

        model._grad(coeffs, grad)
        test = np.array([
            0.4363636, 4.5818182, -0.6009268, 0.4027132, 1.8310919, 0.3308908
        ])
        np.testing.assert_almost_equal(grad, test, decimal=4)
        loss = model._loss_and_grad(coeffs, grad)
        np.testing.assert_almost_equal(grad, test, decimal=4)
        self.assertAlmostEqual(loss, 1.05752053, delta=1e-4)

    def test_model_hawkes_least_sq_serialization(self):
        """...Test that ModelHawkesExpKernLeastSq can be serialized
        """
        for model in [self.model, self.model_list]:
            pickled = pickle.loads(pickle.dumps(model))

            self.assertEqual(model.n_nodes, pickled.n_nodes)
            np.testing.assert_equal(model.decays, pickled.decays)
            self.assertEqual(model.n_jumps, pickled.n_jumps)

            self.assertEqual(model.n_coeffs, pickled.n_coeffs)
            self.assertEqual(model.n_threads, pickled.n_threads)
            np.testing.assert_equal(model.data, pickled.data)
            coeffs = np.random.rand(model.n_coeffs)
            self.assertEqual(model.loss(coeffs), pickled.loss(coeffs))


if __name__ == '__main__':
    unittest.main()
