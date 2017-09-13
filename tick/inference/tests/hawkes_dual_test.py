# License: BSD 3 clause

import unittest
import numpy as np
from tick.simulation import SimuHawkesSumExpKernels

from tick.inference import HawkesDual


class Test(unittest.TestCase):

    def setUp(self):
        np.random.seed(13069)
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.events = [np.cumsum(np.random.rand(10 + i)) for i in range(3)]

        self.decays = np.array([2., 4.])

    @staticmethod
    def get_train_data(n_nodes=3, decays=np.array([1.]), hawkes_seed=13487):
        np.random.seed(130947)
        baseline = np.random.rand(n_nodes) / 4
        adjacency = np.random.rand(n_nodes, n_nodes, len(decays))

        sim = SimuHawkesSumExpKernels(adjacency=adjacency, decays=decays,
                                      baseline=baseline, verbose=False,
                                      seed=hawkes_seed, end_time=3000)
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
        timestamps, baseline, adjacency = Test.get_train_data(n_nodes=n_nodes,
                                                              decays=self.decays)

        l_l2sq = 0.1
        hawkes = HawkesDual(self.decays, l_l2sq, max_iter=150)

        hawkes.fit(timestamps)

        self.assertLess(Test.estimation_error(hawkes.baseline, baseline), 0.1)
        self.assertLess(Test.estimation_error(hawkes.adjacency, adjacency), 0.2)

    def test_hawkes_dual_solution_list_realization(self):
        n_nodes = 3
        timestamps, baseline, adjacency = Test.get_train_data(
            n_nodes=n_nodes, decays=self.decays)
        timestamps_2, _, _ = Test.get_train_data(
            n_nodes=n_nodes, decays=self.decays, hawkes_seed=2039)
        timestamps_list = [timestamps, timestamps_2]

        l_l2sq = 0.1
        hawkes = HawkesDual(self.decays, l_l2sq, max_iter=150)

        hawkes.fit(timestamps_list)

        self.assertLess(Test.estimation_error(hawkes.baseline, baseline), 0.1)
        self.assertLess(Test.estimation_error(hawkes.adjacency, adjacency), 0.2)


if __name__ == '__main__':
    unittest.main()
