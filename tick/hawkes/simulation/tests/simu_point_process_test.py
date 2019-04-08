# License: BSD 3 clause

import unittest

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson, SimuPoissonProcess


class Test(unittest.TestCase):
    def setUp(self):
        self.intensities = [1, 3, 2.2]
        self.run_time = 30
        self.poisson_process = SimuPoissonProcess(
            self.intensities, end_time=self.run_time, verbose=False)

    def test_n_nodes(self):
        """...Test n_nodes attribute of Point Process
        """
        self.assertEqual(self.poisson_process.n_nodes, 3)

    def test_end_time(self):
        """...Test end_time parameter of Point Process
        """
        self.assertEqual(self.poisson_process.end_time, self.run_time)
        self.poisson_process.simulate()
        self.assertEqual(self.poisson_process.end_time, self.run_time)

    def test_simulation_time(self):
        """...Test simulation_time attribute of Point Process
        """
        self.assertEqual(self.poisson_process.simulation_time, 0)
        self.poisson_process.simulate()
        self.assertEqual(self.poisson_process.simulation_time, self.run_time)

    def test_n_total_jumps_and_timestamps(self):
        """...Test n_total_jumps attribute of Point Process
        """
        self.poisson_process.simulate()

        n_total_jumps = sum(map(len, self.poisson_process.timestamps))

        self.assertEqual(self.poisson_process.n_total_jumps, n_total_jumps)

    def test_max_jumps(self):
        """...Test max_jumps attribute of Point Process
        """
        self.poisson_process.end_time = None
        self.poisson_process.max_jumps = 100
        self.assertEqual(self.poisson_process.simulation_time, 0)
        self.assertEqual(self.poisson_process.n_total_jumps, 0)
        self.poisson_process.simulate()
        self.assertEqual(self.poisson_process.n_total_jumps, 100)
        self.assertGreater(self.poisson_process.simulation_time, 0)
        self.assertIsNone(self.poisson_process.end_time)

    def test_is_intensity_tracked(self):
        """...Test is_intensity_tracked method of Point Process
        """
        self.assertFalse(self.poisson_process.is_intensity_tracked())
        self.poisson_process.track_intensity(0.1)
        self.assertTrue(self.poisson_process.is_intensity_tracked())
        self.poisson_process.track_intensity(-0.1)
        self.assertFalse(self.poisson_process.is_intensity_tracked())

    def test_track_intensity(self):
        """...Test that point process intensity is tracked correctly
        """
        t_values = np.linspace(0, self.run_time - 3, 100)
        y_values_1 = np.maximum(0.5 + np.sin(t_values), 0)
        y_values_2 = np.maximum(1. / (1 + t_values), 0)

        tf_1 = TimeFunction((t_values, y_values_1))
        tf_2 = TimeFunction((t_values, y_values_2))

        inhomo_poisson_process = SimuInhomogeneousPoisson(
            [tf_1, tf_2], end_time=self.run_time, seed=2937, verbose=False)

        inhomo_poisson_process.track_intensity(0.1)
        inhomo_poisson_process.simulate()

        tracked_intensity = inhomo_poisson_process.tracked_intensity
        intensity_times = inhomo_poisson_process.intensity_tracked_times

        # Ensure that intensity recorded is equal to the given TimeFunction
        np.testing.assert_array_almost_equal(tracked_intensity[0],
                                             tf_1.value(intensity_times))
        np.testing.assert_array_almost_equal(tracked_intensity[1],
                                             tf_2.value(intensity_times))


if __name__ == "__main__":
    unittest.main()
