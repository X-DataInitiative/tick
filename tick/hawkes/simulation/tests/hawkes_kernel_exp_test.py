# License: BSD 3 clause

import unittest

from tick.hawkes import HawkesKernelExp


class Test(unittest.TestCase):
    def setUp(self):
        self.decay = 2
        self.intensity = 3
        self.hawkes_kernel_exp = HawkesKernelExp(self.intensity, self.decay)

    def test_HawkesKernelExp_decay(self):
        """...Test HawkesKernelExp decay
        """
        self.assertEqual(self.hawkes_kernel_exp.decay, self.decay)

    def test_HawkesKernelExp_intensity(self):
        """...Test HawkesKernelExp intensity
        """
        self.assertEqual(self.hawkes_kernel_exp.intensity, self.intensity)

    def test_HawkesKernelExp_str(self):
        """...Test HawkesKernelExp string representation
        """
        intensity = 3
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str(hawkes_kernel_exp), "3 * 2 * exp(- 2 * t)")

        intensity = 3
        decay = 0
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str(hawkes_kernel_exp), "3")

        intensity = 0
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str(hawkes_kernel_exp), "0")

    def test_HawkesKernelExp_repr(self):
        """...Test HawkesKernelExp string in list representation
        """
        intensity = 3
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str([hawkes_kernel_exp]), "[3*2*exp(-2*t)]")

        intensity = 3
        decay = 0
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str([hawkes_kernel_exp]), "[3]")

        intensity = 0
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(str([hawkes_kernel_exp]), "[0]")

    def test_HawkesKernelExp_strtex(self):
        """...Test HawkesKernelExp latex string representation
        """
        decay = 2
        intensity = 3
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$6 e^{-2 t}$")

        intensity = 3
        decay = 0
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$3$")

        intensity = 0
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$0$")

        intensity = 0.5
        decay = 2
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$e^{-2 t}$")

        decay = 1
        intensity = 3
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$3 e^{- t}$")

        decay = 1
        intensity = 1
        hawkes_kernel_exp = HawkesKernelExp(intensity, decay)
        self.assertEqual(hawkes_kernel_exp.__strtex__(), "$e^{-t}$")


if __name__ == "__main__":
    unittest.main()
