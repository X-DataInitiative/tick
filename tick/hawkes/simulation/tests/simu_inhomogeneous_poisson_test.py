# License: BSD 3 clause

import unittest

import numpy as np

from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson


class Test(unittest.TestCase):
    def test_simulation_1d_inhomogeneous_poisson(self):
        """...Test if simulation of a 1d inhomogeneous Poisson process
        No statistical guarantee on the result"""

        run_time = 30

        t_values = np.linspace(0, run_time - 3, 100)
        y_values = np.maximum(0.5 + np.sin(t_values), 0)

        tf = TimeFunction((t_values, y_values))
        tf_zero = TimeFunction(0)

        inhomo_poisson_process = SimuInhomogeneousPoisson(
            [tf, tf_zero], seed=2937, end_time=run_time, verbose=False)
        inhomo_poisson_process.simulate()

        timestamps = inhomo_poisson_process.timestamps

        # Ensure that non zero TimeFunction intensity ticked at least 2 times
        self.assertGreater(len(timestamps[0]), 2)
        # Ensure that zero TimeFunction intensity never ticked
        self.assertEqual(len(timestamps[1]), 0)

        # Ensure that intensity was non zero when the process did tick
        self.assertEqual(np.prod(tf.value(timestamps[0]) > 0), 1)


if __name__ == "__main__":
    unittest.main()
