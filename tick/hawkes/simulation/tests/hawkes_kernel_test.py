# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes import HawkesKernelExp, HawkesKernelPowerLaw

# We test that HawkesKernel method are correctly instantiated with an
# exponential kernel and a power law kernel


class Test(unittest.TestCase):
    def setUp(self):
        self.decay = 2
        self.intensity = 3
        self.hawkes_kernel_exp = HawkesKernelExp(self.intensity, self.decay)

        self.multiplier = 0.1
        self.cutoff = 0.01
        self.exponent = 1.2
        self.support = 10000
        self.hawkes_kernel_power_law = HawkesKernelPowerLaw(
            self.multiplier, self.cutoff, self.exponent, self.support)

    def test_is_zero(self):
        """...Test is_zero method of HawkesKernel"""
        self.assertFalse(self.hawkes_kernel_exp.is_zero())

    def test_get_support(self):
        """...Test get_support method of HawkesKernel"""
        self.assertEqual(self.hawkes_kernel_power_law.get_support(),
                         self.support)

    def test_get_plot_support(self):
        """...Test get_plot_support method of HawkesKernel"""
        self.assertEqual(self.hawkes_kernel_exp.get_plot_support(),
                         self.intensity / self.decay)

    def test_get_value(self):
        """...Test get_value method of HawkesKernel"""
        self.assertEqual(
            self.hawkes_kernel_exp.get_value(3), 0.014872513059998151)

    def test_get_values(self):
        """...Test get_values method of HawkesKernel"""
        t_values = np.arange(5, dtype=float)
        np.testing.assert_array_almost_equal(
            self.hawkes_kernel_exp.get_values(t_values),
            [6, 8.120117e-01, 1.098938e-01, 1.487251e-02, 2.012776e-03])

    def test_get_norm(self):
        """...Test get_norm method of HawkesKernel"""
        self.assertEqual(self.hawkes_kernel_exp.get_norm(), self.intensity)


if __name__ == "__main__":
    unittest.main()
