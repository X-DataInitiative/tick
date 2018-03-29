# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes import HawkesKernelSumExp


class Test(unittest.TestCase):
    def setUp(self):
        self.decays = np.array([1., 2., 0.2])
        self.intensities = np.array([0.3, 4., 2.])
        self.hawkes_kernel_sumexp = HawkesKernelSumExp(self.intensities,
                                                       self.decays)

    def test_HawkesKernelSumExp_decays(self):
        """...Test HawkesKernelSumExp decays
        """
        np.testing.assert_array_equal(self.hawkes_kernel_sumexp.decays,
                                      self.decays)

    def test_HawkesKernelSumExp_intensities(self):
        """...Test HawkesKernelSumExp intensities
        """
        np.testing.assert_array_equal(self.hawkes_kernel_sumexp.intensities,
                                      self.intensities)

    def test_HawkesKernelSumExp_n_decays(self):
        """...Test HawkesKernelSumExp decay
        """
        self.assertEqual(self.hawkes_kernel_sumexp.n_decays, len(self.decays))

    def test_HawkesKernelSumExp_str(self):
        """...Test HawkesKernelSumExp string representation
        """
        self.assertEqual(
            str(self.hawkes_kernel_sumexp),
            "0.3 * 1 * exp(- 1 * t) + 4 * 2 * exp(- 2 * t) + "
            "2 * 0.2 * exp(- 0.2 * t)")

        self.assertEqual(
            str(self.hawkes_kernel_sumexp),
            "0.3 * 1 * exp(- 1 * t) + 4 * 2 * exp(- 2 * t) + "
            "2 * 0.2 * exp(- 0.2 * t)")

        self.decays[1] = 0
        self.intensities[2] = 0
        hawkes_kernel_sumexp = HawkesKernelSumExp(self.intensities,
                                                  self.decays)
        self.assertEqual(
            str(hawkes_kernel_sumexp), "0.3 * 1 * exp(- 1 * t) + 4 + 0")

    def test_HawkesKernelSumExp_repr(self):
        """...Test HawkesKernelSumExp string in list representation
        """
        self.assertEqual(
            str([self.hawkes_kernel_sumexp]),
            "[0.3*1*exp(-1*t) + 4*2*exp(-2*t) + "
            "2*0.2*exp(-0.2*t)]")

    def test_HawkesKernelSumExp_strtex(self):
        """...Test HawkesKernelSumExp latex string representation
        """
        self.assertEqual(self.hawkes_kernel_sumexp.__strtex__(),
                         "$0.3 e^{- t}$ + $8 e^{-2 t}$ + $0.4 e^{-0.2 t}$")


if __name__ == "__main__":
    unittest.main()
