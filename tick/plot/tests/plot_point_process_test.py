# License: BSD 3 clause

import unittest
import numpy as np

from tick.plot.plot_point_processes import _extract_process_interval


class Test(unittest.TestCase):
    def assert_array_list_equal(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for array1, array2 in zip(list1, list2):
            np.testing.assert_array_almost_equal(array1, array2, decimal=10)

    def test_extract_process_interval(self):
        plot_nodes = range(2)
        end_time = 12.
        original_timestamps = [
            np.linspace(1., 6., 11),
            np.linspace(4., 10., 5)
        ]
        original_intensity_times = np.linspace(0, end_time, 13)
        original_intensities = [
            np.linspace(1., 2.2, 13),
            np.linspace(4.4, 2, 13)
        ]

        # t_min
        timestamps, intensity_times, intensities = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, t_min=5,
            intensity_times=original_intensity_times,
            intensities=original_intensities)
        self.assert_array_list_equal(
            timestamps,
            [np.array([5., 5.5, 6.]),
             np.array([5.5, 7., 8.5, 10.])])
        np.testing.assert_array_equal(intensity_times,
                                      [5., 6., 7., 8., 9., 10., 11., 12.])

        self.assert_array_list_equal(intensities, [
            np.array([1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.1, 2.2]),
            np.array([3.4, 3.2, 3., 2.8, 2.6, 2.4, 2.2, 2.])
        ])

        # t_max
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, t_max=9)
        self.assert_array_list_equal(timestamps, [
            np.array([1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6.]),
            np.array([4., 5.5, 7., 8.5])
        ])

        # t_min and t_max
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, t_min=5, t_max=9)
        self.assert_array_list_equal(
            timestamps, [np.array([5., 5.5, 6.]),
                         np.array([5.5, 7., 8.5])])

        # max_points = 0
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=0)
        self.assert_array_list_equal(timestamps, [np.array([]), np.array([])])

        # max_points
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=2)
        self.assert_array_list_equal(
            timestamps,
            [np.array([1., 1.5]), np.array([])])

        # max_points and t_min
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=2, t_min=3.5)
        self.assert_array_list_equal(
            timestamps,
            [np.array([3.5, 4.]), np.array([4.])])

        # max_points and t_min too big
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=10, t_min=3)
        self.assert_array_list_equal(timestamps, [
            np.array([3., 3.5, 4., 4.5, 5., 5.5, 6.]),
            np.array([4., 5.5, 7., 8.5, 10.])
        ])

        # max_points and t_max
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=4, t_max=8)
        self.assert_array_list_equal(
            timestamps, [np.array([4.5, 5., 5.5, 6.]),
                         np.array([5.5, 7.])])

        # max_points and t_max too small
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, max_jumps=4, t_max=2)
        self.assert_array_list_equal(
            timestamps,
            [np.array([1., 1.5, 2.]), np.array([])])

        # max_points too big, t_min and t_max
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, t_min=5, t_max=9,
            max_jumps=8)
        self.assert_array_list_equal(
            timestamps, [np.array([5., 5.5, 6.]),
                         np.array([5.5, 7., 8.5])])

        # max_points big, t_min and t_max too big
        timestamps, intensity_times, intensity = _extract_process_interval(
            plot_nodes, end_time, original_timestamps, t_min=4, t_max=9,
            max_jumps=3)
        self.assert_array_list_equal(
            timestamps,
            [np.array([4., 4.5, 5.]), np.array([4.])])
