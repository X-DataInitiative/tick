# License: BSD 3 clause

import unittest
import numpy as np
from tick.simulation import SimuHawkesExpKernels

from tick.inference import HawkesDual


class Test(unittest.TestCase):
    def setUp(self):
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2

    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]

        self.decays = 3.

    @staticmethod
    def get_train_data(n_nodes=3, decay=1.):
        np.random.seed(130947)
        baseline = np.random.rand(n_nodes) / 4
        adjacency = np.random.rand(n_nodes, n_nodes)
        if isinstance(decay, (int, float)):
            decay = np.ones((n_nodes, n_nodes)) * decay

        sim = SimuHawkesExpKernels(adjacency=adjacency, decays=decay,
                                   baseline=baseline, verbose=False,
                                   seed=13487, end_time=3000)
        sim.adjust_spectral_radius(0.8)
        adjacency = sim.adjacency
        sim.simulate()

        return sim.timestamps, baseline, adjacency

    @staticmethod
    def estimation_error(estimated, original):
        return np.linalg.norm(original - estimated) ** 2 / \
               np.linalg.norm(original) ** 2

    def test_hawkes_dual_solution(self):
        n_nodes = 3
        decay = 2.4
        timestamps, baseline, adjacency = Test.get_train_data(n_nodes=n_nodes,
                                                              decay=decay)

        l_l2sq = 0.1
        hawkes = HawkesDual(decay, l_l2sq, max_iter=150)

        hawkes.fit(timestamps)

        self.assertLess(Test.estimation_error(hawkes.baseline, baseline), 0.1)
        self.assertLess(Test.estimation_error(hawkes.adjacency, adjacency), 0.1)


if __name__ == '__main__':
    unittest.main()
