# License: BSD 3 clause

import unittest

import itertools
import numpy as np
from tick.base import TimeFunction


def dichotomic_value(t, t_values, y_values,
                     inter_mode=TimeFunction.InterLinear):
    """Returns value of the TimeFunction obtained by dichotomic search

    Parameters
    ----------
    t : `np.ndarray`
        Times at which we compute f(t)

    t_values : `np.ndarray`, shape=(n_points,)
        A sorted array given the time considered

    y_values : `np.ndarray`, shape=(n_points,)
        Array giving the values of the previous t_array

    inter_mode : `TimeFunction.Intermode`
        Selected inter mode

    Returns
    -------
    output : `np.ndarray`
        Interpolated values of f(t) given t_values and y_values
    """
    f_t = np.ones_like(t)

    before_first_time = t < t_values[0]
    after_last_time = t > t_values[-1]

    f_t[before_first_time] = 0
    f_t[after_last_time] = 0

    inside_time = np.invert(before_first_time) & np.invert(after_last_time)

    if inter_mode == TimeFunction.InterLinear:
        f_t[inside_time] = np.interp(t[inside_time], t_values, y_values)

    elif inter_mode == TimeFunction.InterConstLeft:
        index_right = np.searchsorted(t_values, t, side='left')
        f_t[inside_time] = y_values[index_right[inside_time]]

    elif inter_mode == TimeFunction.InterConstRight:
        index_left = np.searchsorted(t_values, t, side='right') - 1
        f_t[inside_time] = y_values[index_left[inside_time]]

    return f_t


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(9372)

        size = 5
        self.t_arrays = [
            np.arange(size, dtype=float),
            np.cumsum(np.random.uniform(0.1, 2, size))
        ]

        self.y_arrays = [
            np.arange(size, dtype=float),
            np.random.uniform(-1, 2, size)
        ]

        self.inter_modes = [
            TimeFunction.InterConstLeft, TimeFunction.InterConstRight,
            TimeFunction.InterLinear
        ]

        self.samples = itertools.product(self.t_arrays, self.y_arrays,
                                         self.inter_modes)

    def test_sample_y(self):
        """...Test that generated sampled_y is correct
        """
        for t_array, y_array, inter_mode in self.samples:
            tf = TimeFunction([t_array, y_array], inter_mode=inter_mode)
            # We remove last value as it is computed but not used
            created_sample_y = tf.sampled_y[:-1]

            sampled_times = t_array[0] + \
                            tf.dt * np.arange(len(created_sample_y))

            true_sample_y = dichotomic_value(sampled_times, t_array, y_array,
                                             inter_mode=inter_mode)

            np.testing.assert_almost_equal(created_sample_y, true_sample_y)

    def test_values(self):
        """...Test that TimeFunction returns correct values on randomly
        selected times
        """
        for t_array, y_array, inter_mode in self.samples:
            # take random t on TimeFunction support
            t_inside_support = np.random.uniform(t_array[0], t_array[-1], 10)
            t_before_support = np.random.uniform(t_array[0] - 4, t_array[0], 5)
            t_after_support = np.random.uniform(t_array[-1], t_array[-1] + 4,
                                                5)

            test_t = np.hstack((t_before_support, t_inside_support,
                                t_after_support, t_array))
            true_values = dichotomic_value(test_t, t_array, y_array,
                                           inter_mode=inter_mode)

            tf = TimeFunction([t_array, y_array], inter_mode=inter_mode)
            tf_values = tf.value(test_t)
            errors = np.abs(true_values - tf_values)

            # If we do not find the same value ensure that the error is
            # controlled by max_error
            different_values = errors > 1e-6

            for t, error in zip(test_t[different_values],
                                errors[different_values]):
                if inter_mode == TimeFunction.InterLinear:
                    self.assertLess(error, tf._max_error(t))
                else:
                    distance_point = np.array(t - t_array).min()
                    self.assertLess(distance_point, tf.dt)

    def test_norm(self):
        """...Test TimeFunction's get_norm method on few known examples
        """
        T = np.array([0, 1, 2], dtype=float)
        Y = np.array([1, 0, -1], dtype=float)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstLeft,
                          dt=0.5)
        self.assertAlmostEqual(tf.get_norm(), -1)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight,
                          dt=0.5)
        self.assertAlmostEqual(tf.get_norm(), 1)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterLinear, dt=0.5)
        self.assertAlmostEqual(tf.get_norm(), 0)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstLeft,
                          dt=0.3)
        self.assertAlmostEqual(tf.get_norm(), -1.1)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterConstRight,
                          dt=0.3)
        self.assertAlmostEqual(tf.get_norm(), 1.2)

        tf = TimeFunction([T, Y], inter_mode=TimeFunction.InterLinear, dt=0.3)
        self.assertAlmostEqual(tf.get_norm(), 0)

    def test_cyclic(self):
        """...Test cyclic border type
        """
        last_value = 2.3
        T = np.linspace(0, last_value)
        Y = np.cos(T * np.pi)

        tf = TimeFunction([T, Y], border_type=TimeFunction.Cyclic)

        self.assertAlmostEqual(
            tf.value(0.3), tf.value(last_value + 0.3), delta=1e-8)


if __name__ == "__main__":
    unittest.main()
