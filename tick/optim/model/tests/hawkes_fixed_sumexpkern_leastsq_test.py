import unittest
import numpy as np
from numpy.linalg import norm

from scipy.optimize import check_grad, fmin_bfgs

from tick.inference.tests.inference import InferenceTest
from tick.optim.model import ModelHawkesFixedSumExpKernLeastSq

from tick.optim.model.tests.hawkes_utils import (
    hawkes_sumexp_kernel_intensities, hawkes_sumexp_kernel_varying_intensities,
    hawkes_least_square_error)


class Test(InferenceTest):
    def setUp(self):
        np.random.seed(30732)

        self.dim = 3
        self.n_realizations = 2
        self.n_decays = 2

        self.decays = np.random.rand(self.n_decays)

        self.timestamps_list = [
            [np.cumsum(np.random.random(np.random.randint(3, 7)))
             for _ in range(self.dim)]
            for _ in range(self.n_realizations)]

        self.baseline = np.random.rand(self.dim)
        self.adjacency = np.random.rand(self.dim, self.dim, self.n_decays)
        self.coeffs = np.hstack((self.baseline, self.adjacency.ravel()))

        self.realization = 0
        self.model = \
            ModelHawkesFixedSumExpKernLeastSq(decays=self.decays)
        self.model.fit(self.timestamps_list[self.realization])

        self.model_list = \
            ModelHawkesFixedSumExpKernLeastSq(decays=self.decays)
        self.model_list.fit(self.timestamps_list)

    def test_model_hawkes_sum_exp_kernel_least_sq_parameters(self):
        """...Test parameters of ModelHawkesFixedSumExpKernLeastSq
        """
        model = ModelHawkesFixedSumExpKernLeastSq([1., 2.])
        self.assertEqual(model.n_decays, 2)
        model = ModelHawkesFixedSumExpKernLeastSq(np.array([1, 2]))
        self.assertEqual(model.n_decays, 2)

        n_baselines = 3
        period_length = 2.
        msg = "n_baselines must be positive"
        with self.assertRaisesRegex(ValueError, msg):
            ModelHawkesFixedSumExpKernLeastSq(self.decays, n_baselines=-1,
                                              period_length=period_length)
        msg = "period_length must be given if multiple baselines are used"
        with self.assertRaisesRegex(ValueError, msg):
            ModelHawkesFixedSumExpKernLeastSq(self.decays,
                                              n_baselines=n_baselines)
        msg = "period_length has no effect when using a constant baseline"
        with self.assertWarnsRegex(UserWarning, msg):
            ModelHawkesFixedSumExpKernLeastSq(self.decays,
                                              period_length=period_length)

    def test_model_hawkes_sum_exp_kernel_least_sq_parameters_change(self):
        """...Test changing parameters of ModelHawkesFixedSumExpKernLeastSq
        """
        model = ModelHawkesFixedSumExpKernLeastSq(self.decays, n_baselines=3,
                                                  period_length=10.)
        new_decays = model.decays + 1.
        model.decays = new_decays
        np.testing.assert_array_equal(model.decays, new_decays)

        new_n_baselines = model.n_baselines + 1
        model.n_baselines = new_n_baselines
        np.testing.assert_array_equal(model.n_baselines, new_n_baselines)

        new_period_length = model.period_length + 1
        model.period_length = new_period_length
        np.testing.assert_array_equal(model.period_length, new_period_length)

    def test_baseline_intervals(self):
        """...Test baseline intervals property of 
        ModelHawkesFixedSumExpKernLeastSq
        """
        n_baselines = 4
        period_length = 10
        model = ModelHawkesFixedSumExpKernLeastSq(decays=self.decays,
                                                  n_baselines=n_baselines,
                                                  period_length=period_length)
        np.testing.assert_array_equal(model.baseline_intervals,
                                      np.array([0., 2.5, 5., 7.5]))

        n_baselines = 1
        model = ModelHawkesFixedSumExpKernLeastSq(decays=self.decays,
                                                  n_baselines=n_baselines)
        np.testing.assert_array_equal(model.baseline_intervals,
                                      np.array([0.]))

    def test_model_hawkes_sum_exp_kernel_least_sq_loss(self):
        """...Test that computed losses are consistent with approximated
        theoretical values
        """
        timestamps = self.timestamps_list[self.realization]

        intensities = hawkes_sumexp_kernel_intensities(
            self.baseline, self.decays, self.adjacency, timestamps)

        integral_approx = hawkes_least_square_error(
            intensities, timestamps,
            self.model.end_times[self.realization])
        integral_approx /= self.model.n_jumps

        self.assertAlmostEqual(integral_approx,
                               self.model.loss(self.coeffs),
                               places=2)

    def test_model_hawkes_least_sq_multiple_events(self):
        """...Test that multiple events list for ModelHawkesFixedExpKernLeastSq
        is consistent with direct integral estimation
        """
        # precision of the integral approximation (and the corresponding
        # tolerance)
        precison = 1

        end_times = np.array([max(map(max, e)) for e in self.timestamps_list])
        end_times += 1.
        self.model_list.fit(self.timestamps_list, end_times=end_times)

        intensities_list = [
            hawkes_sumexp_kernel_intensities(self.baseline, self.decays,
                                             self.adjacency, timestamps)
            for timestamps in self.timestamps_list
        ]

        integral_approx = sum([hawkes_least_square_error(intensities,
                                                         timestamps, end_time,
                                                         precision=precison)
                               for (intensities, timestamps, end_time) in zip(
                intensities_list, self.timestamps_list,
                self.model_list.end_times
            )])

        integral_approx /= self.model_list.n_jumps
        self.assertAlmostEqual(integral_approx,
                               self.model_list.loss(self.coeffs),
                               places=precison)

    def test_model_hawkes_least_sq_incremental_fit(self):
        """...Test that multiple events list for ModelHawkesFixedExpKernLeastSq
        are correctly handle with incremental_fit
        """
        model_incremental_fit = \
            ModelHawkesFixedSumExpKernLeastSq(decays=self.decays)

        for timestamps in self.timestamps_list:
            model_incremental_fit.incremental_fit(timestamps)

        self.assertEqual(model_incremental_fit.loss(self.coeffs),
                         self.model_list.loss(self.coeffs))

    def test_model_hawkes_least_sq_grad(self):
        """...Test that ModelHawkesFixedExpKernLeastSq gradient is consistent
        with loss
        """

        for model in [self.model, self.model_list]:
            self.assertLess(check_grad(model.loss, model.grad, self.coeffs),
                            1e-5)

            # Check that minimum is achievable with a small gradient
            coeffs_min = fmin_bfgs(model.loss, self.coeffs,
                                   fprime=model.grad, disp=False)
            self.assertAlmostEqual(norm(model.grad(coeffs_min)),
                                   .0, delta=1e-4)

    def test_model_hawkes_least_sq_change_decays(self):
        """...Test that loss is still consistent after decays modification in
        ModelHawkesFixedSumExpKernLeastSq
        """
        decays = np.random.rand(self.n_decays)

        self.assertNotEqual(decays[0], self.decays[0])

        model_change_decay = ModelHawkesFixedSumExpKernLeastSq(decays=decays)
        model_change_decay.fit(self.timestamps_list)
        loss_old_decay = model_change_decay.loss(self.coeffs)

        model_change_decay.decays = self.decays

        self.assertNotEqual(loss_old_decay,
                            model_change_decay.loss(self.coeffs))

        self.assertEqual(self.model_list.loss(self.coeffs),
                         model_change_decay.loss(self.coeffs))

    def test_model_hawkes_sum_exp_kernel_varying_baseline_least_sq_loss(self):
        """...Test that computed losses are consistent with approximated
        theoretical values
        """
        timestamps = self.timestamps_list[self.realization]
        end_time = self.model.end_times[self.realization]

        n_baselines = 3
        period_length = 1.
        baselines = np.random.rand(self.dim, n_baselines)

        def baseline_function(i):
            def baseline_value(t):
                first_t = t - period_length * int(t / period_length)
                interval = min(int(first_t / period_length * n_baselines),
                               n_baselines - 1)
                return baselines[i, interval]

            return baseline_value

        baseline_functions = [baseline_function(i) for i in range(self.dim)]

        intensities = hawkes_sumexp_kernel_varying_intensities(
            baseline_functions, self.decays, self.adjacency, timestamps)

        integral_approx = hawkes_least_square_error(
            intensities, timestamps, end_time)
        integral_approx /= self.model.n_jumps

        model = ModelHawkesFixedSumExpKernLeastSq(decays=self.decays,
                                                  n_baselines=n_baselines,
                                                  period_length=period_length)
        model.fit(self.timestamps_list[self.realization])

        coeffs = np.hstack((baselines.ravel(), self.adjacency.ravel()))
        self.assertAlmostEqual(integral_approx,
                               model.loss(coeffs),
                               places=2)

    def test_model_hawkes_varying_baseline_least_sq_grad(self):
        """...Test that ModelHawkesFixedExpKernLeastSq gradient is consistent
        with loss
        """
        for model in [self.model, self.model_list]:
            model.period_length = 1.
            model.n_baselines = 3
            coeffs = np.random.rand(model.n_coeffs)

            self.assertLess(check_grad(model.loss, model.grad, coeffs),
                            1e-5)

            coeffs_min = fmin_bfgs(model.loss, coeffs,
                                   fprime=model.grad, disp=False)

            self.assertAlmostEqual(norm(model.grad(coeffs_min)),
                                   .0, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
