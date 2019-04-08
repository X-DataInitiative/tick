# License: BSD 3 clause

import unittest
import numpy as np
from tick.survival import nelson_aalen, kaplan_meier


class Test(unittest.TestCase):
    def test_hazard_rate_from_survival_function(self):
        n_observations = 100
        timestamps = np.random.uniform(size=n_observations)
        observations = np.ones(n_observations)
        hzrd = nelson_aalen(timestamps, observations)
        surv = kaplan_meier(timestamps, observations)
        surv_from_hzrd = np.exp(-hzrd)
        self.assertTrue(np.allclose(surv, surv_from_hzrd, atol=1e-2))
