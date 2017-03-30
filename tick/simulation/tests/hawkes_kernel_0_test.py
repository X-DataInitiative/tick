import unittest

from tick.simulation.hawkes_kernels import HawkesKernel0


class Test(unittest.TestCase):
    def setUp(self):
        self.hawkes_kernel_0 = HawkesKernel0()

    def test_HawkesKernel0_str(self):
        """...Test HawkesKernel0 string representation
        """
        self.assertEqual(str(self.hawkes_kernel_0), "0")

    def test_HawkesKernel0_repr(self):
        """...Test HawkesKernel0 string in list representation
        """
        self.assertEqual(str([self.hawkes_kernel_0]), "[0]")

    def test_HawkesKernel0_strtex(self):
        """...Test HawkesKernel0 string representation
        """
        self.assertEqual(self.hawkes_kernel_0.__strtex__(), "$0$")


if __name__ == "__main__":
    unittest.main()
