import unittest
import numpy as np
from numpy.linalg import norm

from scipy.optimize import check_grad, fmin_bfgs
from tick.optim.model import ModelHawkesFixedSumExpKernLeastSq

from tick.optim.model.tests.hawkes_utils import \
    hawkes_sumexp_kernel_intensities, \
    hawkes_least_square_error


class Test(unittest.TestCase):
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

    def test_model_hawkes_sum_exp_kearn_least_sq_loss(self):
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


if __name__ == "__main__":
    unittest.main()
