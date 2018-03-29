# License: BSD 3 clause

import unittest

from tick.hawkes import HawkesKernelPowerLaw


class Test(unittest.TestCase):
    def setUp(self):
        self.multiplier = 0.1
        self.cutoff = 0.01
        self.exponent = 1.2
        self.hawkes_kernel_power_law = HawkesKernelPowerLaw(
            self.multiplier, self.cutoff, self.exponent)

    def test_HawkesKernelPowerLaw_multiplier(self):
        """...Test HawkesKernelPowerLaw multiplier
        """
        self.assertEqual(self.hawkes_kernel_power_law.multiplier,
                         self.multiplier)

    def test_HawkesKernelPowerLaw_cutoff(self):
        """...Test HawkesKernelPowerLaw cutoff
        """
        self.assertEqual(self.hawkes_kernel_power_law.cutoff, self.cutoff)

    def test_HawkesKernelPowerLaw_exponent(self):
        """...Test HawkesKernelPowerLaw exponent
        """
        self.assertEqual(self.hawkes_kernel_power_law.exponent, self.exponent)

    def test_HawkesKernelPowerLaw_str(self):
        """...Test HawkesKernelPowerLaw string representation
        """
        multiplier = 0.1
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(
            str(hawkes_kernel_power_law), "0.1 * (0.01 + t)^(-1.2)")

        multiplier = 0
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(str(hawkes_kernel_power_law), "0")

        multiplier = 0.1
        cutoff = 0.01
        exponent = 0
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(str(hawkes_kernel_power_law), "0.1")

    def test_HawkesKernelPowerLaw_repr(self):
        """...Test HawkesKernelPowerLaw string in list representation
        """
        multiplier = 0.1
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(
            str([hawkes_kernel_power_law]), "[0.1*(0.01+t)^(-1.2)]")

        multiplier = 0
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(str([hawkes_kernel_power_law]), "[0]")

        multiplier = 0.1
        cutoff = 0.01
        exponent = 0
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(str([hawkes_kernel_power_law]), "[0.1]")

    def test_HawkesKernelPowerLaw_strtex(self):
        """...Test HawkesKernelPowerLaw latex string representation
        """
        multiplier = 0.1
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(hawkes_kernel_power_law.__strtex__(),
                         "$0.1 (0.01+t)^{-1.2}$")

        multiplier = 0
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(hawkes_kernel_power_law.__strtex__(), "$0$")

        multiplier = 0.1
        cutoff = 0.01
        exponent = 0
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(hawkes_kernel_power_law.__strtex__(), "$0.1$")

        multiplier = 1
        cutoff = 0.01
        exponent = 1.2
        hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff,
                                                       exponent)
        self.assertEqual(hawkes_kernel_power_law.__strtex__(),
                         "$(0.01+t)^{-1.2}$")


if __name__ == "__main__":
    unittest.main()
