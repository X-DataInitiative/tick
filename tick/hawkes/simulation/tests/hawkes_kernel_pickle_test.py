# License: BSD 3 clause

import pickle
import unittest

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import (HawkesKernel0, HawkesKernelExp, HawkesKernelSumExp,
                         HawkesKernelPowerLaw, HawkesKernelTimeFunc)


class Test(unittest.TestCase):
    def setUp(self):
        self.random_times = np.random.rand(10)

    def test_HawkesKernel0_pickle(self):
        """...Test pickling ability of HawkesKernel0
        """
        obj = HawkesKernel0()
        pickled = pickle.loads(pickle.dumps(obj))

        self.assertTrue(str(obj) == str(pickled))
        self.assertEqual(obj.get_support(), pickled.get_support())
        np.testing.assert_array_equal(
            obj.get_values(self.random_times),
            obj.get_values(self.random_times))

    def test_HawkesKernelExp_pickle(self):
        """...Test pickling ability of HawkesKernelExp
        """
        obj = HawkesKernelExp(decay=2, intensity=3)
        pickled = pickle.loads(pickle.dumps(obj))

        self.assertTrue(str(obj) == str(pickled))
        self.assertEqual(obj.decay, pickled.decay)
        self.assertEqual(obj.intensity, pickled.intensity)
        np.testing.assert_array_equal(
            obj.get_values(self.random_times),
            obj.get_values(self.random_times))

    def test_HawkesKernelSumExp_pickle(self):
        """...Test pickling ability of HawkesKernelSumExp
        """
        obj = HawkesKernelSumExp(
            decays=np.arange(1., 2., 0.2), intensities=np.arange(0.3, 2.3, .4))

        pickled = pickle.loads(pickle.dumps(obj))

        self.assertTrue(str(obj) == str(pickled))
        self.assertTrue(np.array_equal(obj.decays, pickled.decays))
        self.assertTrue(np.array_equal(obj.intensities, pickled.intensities))
        np.testing.assert_array_equal(
            obj.get_values(self.random_times),
            obj.get_values(self.random_times))

    def test_HawkesKernelPowerLaw_pickle(self):
        """...Test pickling ability of HawkesKernelPowerLaw
        """
        obj = HawkesKernelPowerLaw(0.1, 0.01, 1.2)
        pickled = pickle.loads(pickle.dumps(obj))

        self.assertTrue(str(obj) == str(pickled))
        np.testing.assert_array_equal(
            obj.get_values(self.random_times),
            obj.get_values(self.random_times))

    def test_HawkesKernelTimeFunc_pickle(self):
        """...Test pickling ability of HawkesKernelTimeFunc
        """
        size = 10
        y_values = np.random.rand(size)
        t_values = np.arange(size, dtype=float)
        time_function = TimeFunction([t_values, y_values])
        obj = HawkesKernelTimeFunc(time_function)

        pickled = pickle.loads(pickle.dumps(obj))

        self.assertEqual(
            obj.time_function.value(1), pickled.time_function.value(1))
        self.assertEqual(
            obj.time_function.value(2), pickled.time_function.value(2))
        self.assertEqual(
            obj.time_function.value(1.5), pickled.time_function.value(1.5))
        self.assertEqual(
            obj.time_function.value(0.75), pickled.time_function.value(0.75))
        np.testing.assert_array_equal(
            obj.get_values(self.random_times),
            obj.get_values(self.random_times))


if __name__ == "__main__":
    unittest.main()
