# License: BSD 3 clause

import unittest
import warnings
from contextlib import contextmanager
from itertools import product

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuHawkes, HawkesKernelExp, \
    HawkesKernelSumExp, \
    HawkesKernel0, HawkesKernelPowerLaw, HawkesKernelTimeFunc, \
    SimuHawkesExpKernels


class Test(unittest.TestCase):
    @contextmanager
    def assertWarnsRegex(self, expected_warning, expected_regex):
        """Reimplement assertWarnsRegex method because Python 3.5 ones is buggy
        """
        with warnings.catch_warnings(record=True) as w:
            yield
            self.assertGreater(len(w), 0, "No warning have been raised")
            self.assertLess(
                len(w), 2, "Several warnings have been raised. "
                "Expected 1")
            self.assertTrue(
                issubclass(w[0].category, expected_warning),
                'Expected %s got %s' % (expected_warning, w[0].category))
            self.assertRegex(
                str(w[0].message), expected_regex,
                "Warnings regex do not match")

    def setUp(self):
        np.random.seed(28374)

        self.kernels = np.array([[HawkesKernel0(),
                                  HawkesKernelExp(0.1, 3)], [
                                      HawkesKernelPowerLaw(0.2, 4, 2),
                                      HawkesKernelSumExp([0.1, 0.4], [3, 4])
                                  ]])

        t_values = np.linspace(0, 10, 10)
        y_values = np.maximum(0.5 + np.sin(t_values), 0)
        self.time_func_kernel = HawkesKernelTimeFunc(t_values=t_values,
                                                     y_values=y_values)

        self.baseline = np.random.rand(2)

    def test_hawkes_set_kernel(self):
        """...Test Hawkes process kernels can be set after initialization
        """
        hawkes = SimuHawkes(n_nodes=2)

        for i, j in product(range(2), range(2)):
            hawkes.set_kernel(i, j, self.kernels[i, j])

        for i, j in product(range(2), range(2)):
            self.assertEqual(hawkes.kernels[i, j], self.kernels[i, j])

        hawkes.set_kernel(1, 1, self.time_func_kernel)
        self.assertEqual(hawkes.kernels[1, 1], self.time_func_kernel)

    def test_hawkes_set_baseline_piecewiseconstant(self):
        """...Test Hawkes process baseline set with time and value arrays
        """
        baselines = [[1., 2., 1.5, 4.], [2., 1.5, 4., 1.]]
        hawkes = SimuHawkes(baseline=baselines, period_length=3.5,
                            kernels=self.kernels, verbose=False)

        hawkes.end_time = 10
        hawkes.simulate()
        self.assertGreater(hawkes.n_total_jumps, 1)

    def test_hawkes_set_baseline_timefunction(self):
        """...Test Hawkes process baseline set with TimeFunction
        """
        t_values = [0.5, 1., 2., 3.5]
        y_values_1 = [1., 2., 1.5, 4.]
        y_values_2 = [2., 1.5, 4., 1.]
        timefunction1 = TimeFunction((t_values, y_values_1))
        timefunction2 = TimeFunction((t_values, y_values_2))
        hawkes = SimuHawkes(baseline=[timefunction1, timefunction2],
                            kernels=self.kernels, verbose=False)
        hawkes.end_time = 10
        hawkes.simulate()
        self.assertGreater(hawkes.n_total_jumps, 1)

    def test_hawkes_mean_intensity(self):
        """...Test that Hawkes obtained mean intensity is consistent
        """

        hawkes = SimuHawkes(kernels=self.kernels, baseline=self.baseline,
                            seed=308, end_time=300, verbose=False)
        self.assertLess(hawkes.spectral_radius(), 1)

        hawkes.track_intensity(0.01)
        hawkes.simulate()

        mean_intensity = hawkes.mean_intensity()
        for i in range(hawkes.n_nodes):
            self.assertAlmostEqual(
                np.mean(hawkes.tracked_intensity[i]), mean_intensity[i],
                delta=0.3)

    def test_simu_hawkes_constructor(self):
        """...Test SimuHawkes constructor
        """
        hawkes = SimuHawkes(kernels=self.kernels)
        self.assertEqual(hawkes.n_nodes, 2)
        np.testing.assert_array_equal(hawkes.baseline, np.zeros(2))
        for i, j in product(range(2), range(2)):
            self.assertEqual(hawkes.kernels[i, j], self.kernels[i, j])

        hawkes = SimuHawkes(baseline=self.baseline)
        self.assertEqual(hawkes.n_nodes, 2)
        np.testing.assert_array_equal(hawkes.baseline, self.baseline)
        for i, j in product(range(2), range(2)):
            self.assertEqual(hawkes.kernels[i, j].__class__, HawkesKernel0)
            self.assertEqual(hawkes.kernels[i, j], hawkes._kernel_0)

        hawkes = SimuHawkes(n_nodes=2)
        self.assertEqual(hawkes.n_nodes, 2)
        np.testing.assert_array_equal(hawkes.baseline, np.zeros(2))
        for i, j in product(range(2), range(2)):
            self.assertEqual(hawkes.kernels[i, j].__class__, HawkesKernel0)
            self.assertEqual(hawkes.kernels[i, j], hawkes._kernel_0)

    def test_simu_hawkes_constructor_errors(self):
        """...Test error messages raised by SimuHawkes constructor
        """
        bad_baseline = np.random.rand(4)
        bad_kernels = self.kernels[:, 0:1]

        msg = '^kernels and baseline have different length. kernels has ' \
              'length 2, whereas baseline has length 4\.$'
        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes(kernels=self.kernels, baseline=bad_baseline)

        msg = "^n_nodes will be automatically calculated if baseline or " \
              "kernels is set$"
        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes(kernels=self.kernels, n_nodes=2)

        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes(baseline=self.baseline, n_nodes=2)

        msg = "^n_nodes must be given if neither kernels, nor baseline are " \
              "given$"
        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes()

        msg = '^kernels shape should be \(2, 2\) instead of \(2, 1\)$'
        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes(kernels=bad_kernels)

        msg = '^n_nodes must be positive but equals -1$'
        with self.assertRaisesRegex(ValueError, msg):
            SimuHawkes(n_nodes=-1)

    def test_simu_hawkes_force_simulation(self):
        """...Test force_simulation parameter of SimuHawkes
        """
        diverging_kernel = [[HawkesKernelExp(2, 3)]]
        hawkes = SimuHawkes(kernels=diverging_kernel, baseline=[1],
                            verbose=False)
        hawkes.end_time = 10

        msg = '^Simulation not launched as this Hawkes process is not ' \
              'stable \(spectral radius of 2\). You can use ' \
              'force_simulation parameter if you really want to simulate it$'
        with self.assertRaisesRegex(ValueError, msg):
            hawkes.simulate()

        msg = "^This process has already be simulated until time 0.000000$"
        with self.assertWarnsRegex(UserWarning, msg):
            hawkes.end_time = 0
            hawkes.force_simulation = True
            hawkes.simulate()

    def test_hawkes_negative_intensity_fail(self):
        """...Test simulation with negative kernel without threshold_negative_intensity
        """
        run_time = 40

        hawkes = SimuHawkes(n_nodes=1, end_time=run_time, verbose=False,
                            seed=1398)
        kernel = HawkesKernelExp(-1.3, .8)
        hawkes.set_kernel(0, 0, kernel)
        hawkes.set_baseline(0, 0.3)

        msg = 'Simulation stopped because intensity went negative ' \
              '\(you could call ``threshold_negative_intensity`` to allow it\)'
        with self.assertRaisesRegex(RuntimeError, msg):
            hawkes.simulate()

    def test_hawkes_negative_intensity(self):
        """...Test simulation with negative kernel
        """
        run_time = 40

        hawkes = SimuHawkes(n_nodes=1, end_time=run_time, verbose=False,
                            seed=1398)
        kernel = HawkesKernelExp(-1.3, .8)
        hawkes.set_kernel(0, 0, kernel)
        hawkes.set_baseline(0, 0.3)
        hawkes.threshold_negative_intensity()

        dt = 0.1
        hawkes.track_intensity(dt)
        hawkes.simulate()

        self.assertAlmostEqual(hawkes.tracked_intensity[0].min(), 0)
        self.assertAlmostEqual(hawkes.tracked_intensity[0].max(),
                               hawkes.baseline[0])
        self.assertGreater(hawkes.n_total_jumps, 1)

    def test_hawkes_set_timestamps(self):
        """...Test simulation after some timestamps have been set manually
        """
        decay = 0.5
        baseline = [0.2, 0.4]
        adjacency = [[0.2, 0.1], [0, 0.3]]

        hawkes = SimuHawkesExpKernels(baseline=baseline, decays=decay,
                                      adjacency=adjacency, verbose=False,
                                      seed=1393)

        # timestamps generated by a hawkes process with the same
        # characteristics simulated up to time 10
        original_timestamps = [
            np.array([7.096244, 9.389927]),
            np.array([
                0.436199, 0.659153, 2.622352, 3.095093, 7.189881, 8.068153,
                9.240032
            ]),
        ]

        hawkes.track_intensity(1)
        hawkes.set_timestamps(original_timestamps, 10)
        hawkes.end_time = 100
        hawkes.simulate()

        # Intensity up to time 10 is known
        first_intensities = hawkes.tracked_intensity[0][:10]
        np.testing.assert_array_almost_equal(first_intensities, [
            0.2, 0.2447256, 0.27988282, 0.24845138, 0.23549475, 0.27078386,
            0.26749709, 0.27473586, 0.24532959, 0.22749379
        ])

        # Ensure other jumps have occured afterwards
        self.assertGreater(hawkes.n_total_jumps,
                           sum(map(len, original_timestamps)))


if __name__ == "__main__":
    unittest.main()
