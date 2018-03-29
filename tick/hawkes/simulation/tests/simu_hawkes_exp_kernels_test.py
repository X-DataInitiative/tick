# License: BSD 3 clause

import unittest
from itertools import product

import numpy as np

from tick.hawkes import SimuHawkesExpKernels, HawkesKernelExp, HawkesKernel0


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(23982)
        self.n_nodes = 3
        self.baseline = np.random.rand(self.n_nodes)
        self.adjacency = np.random.rand(self.n_nodes, self.n_nodes) / 2
        self.decays = np.random.rand(self.n_nodes, self.n_nodes)

        self.adjacency[0, 0] = 0
        self.adjacency[-1, -1] = 0

        self.hawkes = SimuHawkesExpKernels(self.adjacency, self.decays,
                                           baseline=self.baseline, seed=203,
                                           verbose=False)

    def test_hawkes_exponential_kernels(self):
        """...Test creation of a Hawkes Process with exponential kernels
        """

        kernel_0 = None
        for i, j in product(range(self.n_nodes), range(self.n_nodes)):
            kernel_ij = self.hawkes.kernels[i, j]

            if self.adjacency[i, j] == 0:
                self.assertEqual(kernel_ij.__class__, HawkesKernel0)

                # We check that all 0 adjacency share the same kernel 0
                # This might save lots of memory with very large,
                # very sparse adjacency matrices
                if kernel_0 is None:
                    kernel_0 = kernel_ij
                else:
                    self.assertEqual(kernel_0, kernel_ij)

            else:
                self.assertEqual(kernel_ij.__class__, HawkesKernelExp)
                self.assertEqual(kernel_ij.decay, self.decays[i, j])
                self.assertEqual(kernel_ij.intensity, self.adjacency[i, j])

        np.testing.assert_array_equal(self.baseline, self.hawkes.baseline)

    def test_hawkes_spectral_radius_exp_kernel(self):
        """...Hawkes Process spectral radius and adjust spectral radius
        methods
        """
        self.assertAlmostEqual(self.hawkes.spectral_radius(),
                               0.6645446549735008)

        self.hawkes.adjust_spectral_radius(0.6)
        self.assertAlmostEqual(self.hawkes.spectral_radius(), 0.6)

    def test_hawkes_mean_intensity(self):
        """...Test that Hawkes obtained mean intensity is consistent
        """

        self.assertLess(self.hawkes.spectral_radius(), 1)

        self.hawkes.end_time = 1000
        self.hawkes.track_intensity(0.01)
        self.hawkes.simulate()

        mean_intensity = self.hawkes.mean_intensity()
        for i in range(self.hawkes.n_nodes):
            self.assertAlmostEqual(
                np.mean(self.hawkes.tracked_intensity[i]), mean_intensity[i],
                delta=0.1)


if __name__ == "__main__":
    unittest.main()
