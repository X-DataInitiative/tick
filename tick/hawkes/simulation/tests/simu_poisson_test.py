# License: BSD 3 clause

import unittest

import numpy as np
from scipy.stats import norm

from tick.hawkes import SimuPoissonProcess


class Test(unittest.TestCase):
    def setUp(self):
        # Result of each coordinate would be out the confidence interval
        # with this probability
        confidence = 1e-4
        self.z = norm.ppf(1 - confidence / 2)

    def test_poisson_constructor(self):
        """...Test constructor of Poisson process
        """
        intensity = 2.9
        poisson_process = SimuPoissonProcess(intensity)
        self.assertEqual(poisson_process.intensities, intensity)

        intensities = np.array([1.0, 2.0, 3.3])
        poisson_process = SimuPoissonProcess(intensities)
        np.testing.assert_array_equal(poisson_process.intensities, intensities)

    def test_simulation_1d_poisson(self):
        """...Test for simulation of a Poisson Process of one dimension
        We check that the number of jumps is in the correct confidence interval
        """

        # First in one dimension
        lambda_0 = 2.9
        time = 1000.0

        poisson_process = SimuPoissonProcess(lambda_0, seed=139, end_time=time,
                                             verbose=False)
        poisson_process.simulate()
        n_total_jumps = poisson_process.n_total_jumps
        tcl = (n_total_jumps - time * lambda_0) / np.sqrt(time * lambda_0)

        self.assertLess(np.abs(tcl), self.z)

    def test_simulation_nd_poisson(self):
        """...Test for simulation of a multidimensional Poisson Process
        We check that the number of jumps is in the correct confidence interval
        """
        lambdas = np.array([1.0, 2.0, 3.3])
        time = 1000.0

        poi = SimuPoissonProcess(lambdas, seed=13923, end_time=time,
                                 verbose=False)
        poi.simulate()

        timestamps = poi.timestamps
        jumps = np.array(list(map(len, timestamps)))
        tcl = np.divide(jumps - time * lambdas, np.sqrt(time * lambdas))

        n_fails = sum(np.abs(tcl) > self.z)
        self.assertEqual(n_fails, 0)


if __name__ == "__main__":
    unittest.main()
