# License: BSD 3 clause

import unittest

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc


class Test(unittest.TestCase):
    def setUp(self):
        size = 10
        self.y_values = np.random.rand(size)
        self.t_values = np.arange(size, dtype=float)
        self.time_function = TimeFunction([self.t_values, self.y_values])
        self.hawkes_kernel_time_func = HawkesKernelTimeFunc(self.time_function)

    def test_HawkesKernelTimeFunc_time_function(self):
        """...Test HawkesKernelTimeFunc time_function parameter
        """
        t_values = np.linspace(-1, 100, 1000)
        np.testing.assert_array_equal(
            self.hawkes_kernel_time_func.time_function.value(t_values),
            self.time_function.value(t_values))

        self.hawkes_kernel_time_func = \
            HawkesKernelTimeFunc(t_values=self.t_values, y_values=self.y_values)
        np.testing.assert_array_equal(
            self.hawkes_kernel_time_func.time_function.value(t_values),
            self.time_function.value(t_values))

    def test_HawkesKernelTimeFunc_str(self):
        """...Test HawkesKernelTimeFunc string representation
        """
        self.assertEqual(str(self.hawkes_kernel_time_func), "KernelTimeFunc")

    def test_HawkesKernelTimeFunc_repr(self):
        """...Test HawkesKernelTimeFunc string in list representation
        """
        self.assertEqual(
            str([self.hawkes_kernel_time_func]), "[KernelTimeFunc]")

    def test_HawkesKernelTimeFunc_strtex(self):
        """...Test HawkesKernelTimeFunc string representation
        """
        self.assertEqual(self.hawkes_kernel_time_func.__strtex__(),
                         "TimeFunc Kernel")


if __name__ == "__main__":
    unittest.main()
