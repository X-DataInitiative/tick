# License: BSD 3 clause

import unittest

import numpy as np
from scipy.optimize import check_grad

from tick.hawkes.model import ModelHawkesExpKernLogLik
from tick.hawkes.model.tests.model_hawkes_test_utils import \
    hawkes_log_likelihood, \
    hawkes_exp_kernel_intensities


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(30732)

        self.n_nodes = 3
        self.n_realizations = 2

        self.decay = np.random.rand()

        self.timestamps_list = [[
            np.cumsum(np.random.random(np.random.randint(3, 7)))
            for _ in range(self.n_nodes)
        ] for _ in range(self.n_realizations)]

        self.end_time = 10

        self.baseline = np.random.rand(self.n_nodes)
        self.adjacency = np.random.rand(self.n_nodes, self.n_nodes)
        self.coeffs = np.hstack((self.baseline, self.adjacency.ravel()))

        self.realization = 0
        self.model = ModelHawkesExpKernLogLik(self.decay)
        self.model.fit(self.timestamps_list[self.realization],
                       end_times=self.end_time)

        self.model_list = ModelHawkesExpKernLogLik(self.decay)
        self.model_list.fit(self.timestamps_list)

    def test_model_hawkes_losses(self):
        """...Test that computed losses are consistent with approximated
        theoretical values
        """
        timestamps = self.timestamps_list[self.realization]

        decays = np.ones((self.n_nodes, self.n_nodes)) * self.decay
        intensities = hawkes_exp_kernel_intensities(self.baseline, decays,
                                                    self.adjacency, timestamps)

        precision = 3
        integral_approx = hawkes_log_likelihood(
            intensities, timestamps, self.end_time, precision=precision)
        integral_approx /= self.model.n_jumps

        self.assertAlmostEqual(integral_approx, -self.model.loss(self.coeffs),
                               places=precision)

    def test_model_hawkes_loglik_multiple_events(self):
        """...Test that multiple events list for ModelHawkesExpKernLogLik
        is consistent with direct integral estimation
        """
        end_times = np.array([max(map(max, e)) for e in self.timestamps_list])
        end_times += 1.
        self.model_list.fit(self.timestamps_list, end_times=end_times)

        decays = np.ones((self.n_nodes, self.n_nodes)) * self.decay
        intensities_list = [
            hawkes_exp_kernel_intensities(self.baseline, decays,
                                          self.adjacency, timestamps)
            for timestamps in self.timestamps_list
        ]

        integral_approx = sum([
            hawkes_log_likelihood(intensities, timestamps, end_time)
            for (intensities, timestamps,
                 end_time) in zip(intensities_list, self.timestamps_list,
                                  self.model_list.end_times)
        ])

        integral_approx /= self.model_list.n_jumps
        self.assertAlmostEqual(integral_approx,
                               -self.model_list.loss(self.coeffs), places=2)

    def test_model_hawkes_loglik_incremental_fit(self):
        """...Test that multiple events list for ModelHawkesExpKernLogLik
        are correctly handle with incremental_fit
        """
        model_incremental_fit = ModelHawkesExpKernLogLik(decay=self.decay)

        for timestamps in self.timestamps_list:
            model_incremental_fit.incremental_fit(timestamps)

        self.assertEqual(
            model_incremental_fit.loss(self.coeffs),
            self.model_list.loss(self.coeffs))

    def test_model_hawkes_loglik_grad(self):
        """...Test that ModelHawkesExpKernLeastSq gradient is consistent
        with loss
        """
        self.assertLess(
            check_grad(self.model.loss, self.model.grad, self.coeffs), 1e-5)

    def test_model_hawkes_loglik_hessian_norm(self):
        """...Test that ModelHawkesExpKernLeastSq hessian norm is
        consistent with gradient
        """
        self.assertLess(
            check_grad(self.model.loss, self.model.grad, self.coeffs), 1e-5)

    def test_hawkesgrad_hess_norm(self):
        """...Test if grad and log likelihood are correctly computed
        """
        hessian_point = np.random.rand(self.model.n_coeffs)
        vector = np.random.rand(self.model.n_coeffs)

        hessian_norm = self.model.hessian_norm(hessian_point, vector)

        delta = 1e-7
        grad_point_minus = self.model.grad(hessian_point + delta * vector)
        grad_point_plus = self.model.grad(hessian_point - delta * vector)
        finite_diff_result = vector.dot(grad_point_minus - grad_point_plus)
        finite_diff_result /= (2 * delta)
        self.assertAlmostEqual(finite_diff_result, hessian_norm)

        hessian_result = vector.T.dot(
            self.model.hessian(hessian_point).dot(vector))
        self.assertAlmostEqual(hessian_result, hessian_norm)

    def test_model_hawkes_loglik_change_decays(self):
        """...Test that loss is still consistent after decays modification in
        ModelHawkesExpKernLogLik
        """
        decay = np.random.rand()

        self.assertNotEqual(decay, self.decay)

        model_change_decay = ModelHawkesExpKernLogLik(decay=decay)
        model_change_decay.fit(self.timestamps_list)
        loss_old_decay = model_change_decay.loss(self.coeffs)

        model_change_decay.decay = self.decay

        self.assertNotEqual(loss_old_decay,
                            model_change_decay.loss(self.coeffs))

        self.assertEqual(
            self.model_list.loss(self.coeffs),
            model_change_decay.loss(self.coeffs))

    def test_hawkes_list_n_threads(self):
        """...Test that the number of used threads is as expected
        """
        model_list = ModelHawkesExpKernLogLik(decay=self.decay, n_threads=1)

        # 0 threads yet as no data has been given
        self.assertEqual(model_list._model.get_n_threads(), 0)

        # Now that it has been fitted it equals
        # min(n_threads, n_nodes * n_realizations)
        model_list.fit(self.timestamps_list)
        self.assertEqual(model_list._model.get_n_threads(), 1)

        model_list.n_threads = 8
        self.assertEqual(model_list._model.get_n_threads(), 6)

        realization_2_nodes = [np.array([3., 4.]), np.array([3.5, 6.])]

        model_list.fit(realization_2_nodes)
        self.assertEqual(model_list._model.get_n_threads(), 2)

        model_list.n_threads = 1
        self.assertEqual(model_list._model.get_n_threads(), 1)

    def test_ModelHawkesExpKernLogLik_hessian(self):
        """...Numerical consistency check of hessian for Hawkes loglik
        """
        for model in [self.model]:
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
                    h = model.hessian(x).todense()
                    return np.asarray(h)[i, :]

                self.assertLess(check_grad(g_i, h_i, self.coeffs), 1e-5)

    def test_ModelHawkesExpKernLogLik_refit(self):
        model = ModelHawkesExpKernLogLik(decay=1.0)
        model.fit(events=[np.array([0.0, 50.0])], end_times=100.0)
        model.fit(events=[np.array([0.0, 500.0])], end_times=1000.0)
        model.fit(events=[np.array([0.0, 50.0])], end_times=100.0)


if __name__ == '__main__':
    unittest.main()
