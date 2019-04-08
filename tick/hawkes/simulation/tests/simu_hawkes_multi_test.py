# License: BSD 3 clause

import unittest

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import (SimuHawkes, HawkesKernelExp, HawkesKernelSumExp,
                         HawkesKernel0, HawkesKernelPowerLaw,
                         HawkesKernelTimeFunc, SimuHawkesMulti)


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(28374)

        self.kernels = np.array([[HawkesKernel0(),
                                  HawkesKernelExp(0.1, 3)], [
                                      HawkesKernelPowerLaw(0.2, 4, 2),
                                      HawkesKernelSumExp([0.1, 0.4], [3, 4])
                                  ]])

        self.baseline = np.random.rand(2)

    def test_simu_hawkes_multi_attrs(self):
        """...Test multiple simulations via SimuHawkesMulti vs. single Hawkes

        See that multiple simulations has same attributes as a single Hawkes
        simulation, but different results
        """

        hawkes = SimuHawkes(kernels=self.kernels, baseline=self.baseline,
                            end_time=10, verbose=False, seed=504)

        multi = SimuHawkesMulti(hawkes, n_threads=4, n_simulations=10)
        multi.simulate()

        hawkes.simulate()

        np.testing.assert_array_equal(hawkes.simulation_time,
                                      multi.simulation_time)
        np.testing.assert_array_equal(hawkes.n_nodes, multi.n_nodes)
        np.testing.assert_array_equal(hawkes.end_time, multi.end_time)
        np.testing.assert_array_equal(hawkes.max_jumps, multi.max_jumps)
        np.testing.assert_array_equal(hawkes.spectral_radius(),
                                      multi.spectral_radius)

        self.assertTrue(
            all(
                np.array_equal(hawkes.mean_intensity(), np.array(x))
                for x in multi.mean_intensity))

        self.assertFalse(
            np.array_equal(hawkes.n_total_jumps, multi.n_total_jumps))

    def test_simu_hawkes_multi_seed(self):
        """...Test seeded Hawkes simu is re-seeded under multiple simulations
        """
        seed = 504
        hawkes = SimuHawkes(kernels=self.kernels, baseline=self.baseline,
                            end_time=10, verbose=False)

        seeded_hawkes = SimuHawkes(kernels=self.kernels,
                                   baseline=self.baseline, end_time=10,
                                   verbose=False, seed=seed)

        multi_1 = SimuHawkesMulti(seeded_hawkes, n_threads=4, n_simulations=1)
        multi_2 = SimuHawkesMulti(hawkes, n_threads=4, n_simulations=1)
        multi_2.seed = seed

        hawkes.seed = seed
        multi_3 = SimuHawkesMulti(hawkes, n_threads=4, n_simulations=1)

        multi_1.simulate()
        multi_2.simulate()
        multi_3.simulate()

        np.testing.assert_array_equal(multi_1.n_total_jumps,
                                      multi_2.n_total_jumps)
        np.testing.assert_array_equal(multi_1.n_total_jumps,
                                      multi_3.n_total_jumps)

        timestamps_1 = multi_1.timestamps
        timestamps_2 = multi_2.timestamps
        timestamps_3 = multi_3.timestamps

        self.assertEqual(len(timestamps_1), len(timestamps_2))
        self.assertEqual(len(timestamps_1), len(timestamps_3))

        for (t1, t2, t3) in zip(timestamps_1, timestamps_2, timestamps_3):
            np.testing.assert_array_equal(t1[0], t2[0])
            np.testing.assert_array_equal(t1[1], t2[1])
            np.testing.assert_array_equal(t1[0], t3[0])
            np.testing.assert_array_equal(t1[1], t3[1])

    def test_simu_hawkes_no_seed(self):
        """...Test hawkes multi can be simulated even if no seed is given
        """
        T1 = np.array([0, 2, 2.5], dtype=float)
        Y1 = np.array([0, .6, 0], dtype=float)
        tf = TimeFunction([T1, Y1], inter_mode=TimeFunction.InterConstRight,
                          dt=0.1)
        kernel = HawkesKernelTimeFunc(tf)
        hawkes = SimuHawkes(baseline=[.1], end_time=100, verbose=False)
        hawkes.set_kernel(0, 0, kernel)
        multi_hawkes_1 = SimuHawkesMulti(hawkes, n_simulations=5)
        multi_hawkes_1.simulate()

        multi_hawkes_2 = SimuHawkesMulti(hawkes, n_simulations=5)
        multi_hawkes_2.simulate()

        # If no seed are given, realizations must be different
        self.assertNotEqual(multi_hawkes_1.timestamps[0][0][0],
                            multi_hawkes_2.timestamps[0][0][0])

    def test_simu_hawkes_multi_time_func(self):
        """...Test that hawkes multi works correctly with HawkesKernelTimeFunc
        """
        run_time = 100

        t_values1 = np.array([0, 1, 1.5], dtype=float)
        y_values1 = np.array([0, .2, 0], dtype=float)
        tf1 = TimeFunction([t_values1, y_values1],
                           inter_mode=TimeFunction.InterConstRight, dt=0.1)
        kernel1 = HawkesKernelTimeFunc(tf1)

        t_values2 = np.array([0, 2, 2.5], dtype=float)
        y_values2 = np.array([0, .6, 0], dtype=float)
        tf2 = TimeFunction([t_values2, y_values2],
                           inter_mode=TimeFunction.InterConstRight, dt=0.1)
        kernel2 = HawkesKernelTimeFunc(tf2)

        baseline = np.array([0.1, 0.3])

        hawkes = SimuHawkes(baseline=baseline, end_time=run_time,
                            verbose=False, seed=2334)

        hawkes.set_kernel(0, 0, kernel1)
        hawkes.set_kernel(0, 1, kernel1)
        hawkes.set_kernel(1, 0, kernel2)
        hawkes.set_kernel(1, 1, kernel2)

        hawkes_multi = SimuHawkesMulti(hawkes, n_simulations=5, n_threads=4)
        hawkes_multi.simulate()


if __name__ == "__main__":
    unittest.main()
