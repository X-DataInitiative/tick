# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti
from tick.hawkes.inference import HawkesADM4


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(329832)
        self.decay = 0.7
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2

    def simulate_sparse_realization(self):
        """Simulate realization in which some nodes are sometimes empty
        """
        baseline = np.array([0.3, 0.001])
        adjacency = np.array([[0.5, 0.8], [0., 1.3]])

        sim = SimuHawkesExpKernels(adjacency=adjacency, decays=self.decay,
                                   baseline=baseline, verbose=False,
                                   seed=13487, end_time=500)
        sim.adjust_spectral_radius(0.8)
        multi = SimuHawkesMulti(sim, n_simulations=100)

        adjacency = sim.adjacency
        multi.simulate()

        # Check that some but not all realizations are empty
        self.assertGreater(max(map(lambda r: len(r[1]), multi.timestamps)), 1)
        self.assertEqual(min(map(lambda r: len(r[1]), multi.timestamps)), 0)

        return baseline, adjacency, multi.timestamps

    def test_sparse(self):
        """...Test that original coeffs are correctly retrieved when some
        realizations are empty
        """
        baseline, adjacency, events = self.simulate_sparse_realization()

        learner = HawkesADM4(self.decay, verbose=False)
        learner.fit(events)

        np.testing.assert_array_almost_equal(learner.baseline, baseline,
                                             decimal=1)
        np.testing.assert_array_almost_equal(learner.adjacency, adjacency,
                                             decimal=1)

    def test_hawkes_adm4_solution(self):
        """...Test solution obtained by HawkesADM4 on toy timestamps
        """
        events = [[
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ], [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19])
        ]]

        n_nodes = len(events[0])
        rho = 0.5
        C = 10
        lasso_nuclear_ratio = 0.7

        baseline_start = np.zeros(n_nodes) + .2
        adjacency_start = np.zeros((n_nodes, n_nodes)) + .2

        learner = HawkesADM4(self.decay, rho=rho, C=C,
                             lasso_nuclear_ratio=lasso_nuclear_ratio,
                             n_threads=3, max_iter=11, verbose=False,
                             em_max_iter=3, record_every=1)
        learner.fit(events[0], baseline_start=baseline_start,
                    adjacency_start=adjacency_start)

        baseline = np.array([0.14551, 0.239859])

        adjacency = np.array([[2.275416e-01, 8.234672e-02],
                              [1.195861e-02, 4.070548e-10]])

        np.testing.assert_array_almost_equal(learner.baseline, baseline,
                                             decimal=6)
        np.testing.assert_array_almost_equal(learner.adjacency, adjacency,
                                             decimal=6)

    def test_hawkes_adm4_score(self):
        """...Test HawkesADM4 score method
        """
        n_nodes = 2
        n_realizations = 3

        train_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        test_events = [[
            np.cumsum(np.random.rand(4 + i)) for i in range(n_nodes)
        ] for _ in range(n_realizations)]

        learner = HawkesADM4(self.decay, record_every=1)

        msg = '^You must either call `fit` before `score` or provide events$'
        with self.assertRaisesRegex(ValueError, msg):
            learner.score()

        given_baseline = np.random.rand(n_nodes)
        given_adjacency = np.random.rand(n_nodes, n_nodes)

        learner.fit(train_events)

        train_score_current_coeffs = learner.score()
        self.assertAlmostEqual(train_score_current_coeffs, 0.12029826)

        train_score_given_coeffs = learner.score(baseline=given_baseline,
                                                 adjacency=given_adjacency)
        self.assertAlmostEqual(train_score_given_coeffs, -0.15247511)

        test_score_current_coeffs = learner.score(test_events)
        self.assertAlmostEqual(test_score_current_coeffs, 0.17640007)

        test_score_given_coeffs = learner.score(
            test_events, baseline=given_baseline, adjacency=given_adjacency)
        self.assertAlmostEqual(test_score_given_coeffs, -0.07973875)

    def test_hawkes_adm4_set_data(self):
        """...Test set_data method of Hawkes ADM4
        """
        events = [[
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ], [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19])
        ]]

        learner = HawkesADM4(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        events = [
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ]

        learner = HawkesADM4(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        msg = "All realizations should have 2 nodes, but realization 1 has " \
              "1 nodes"
        with self.assertRaisesRegex(RuntimeError, msg):
            events = [[
                np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
                np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
            ], [np.array([2, 3.2, 11.4, 12.8, 45])]]
            learner._set_data(events)

    def test_hawkes_adm4_parameters(self):
        """...Test that hawkes adm4 parameters are correctly linked
        """
        learner = HawkesADM4(self.float_1)
        self.assertEqual(learner.decay, self.float_1)
        self.assertEqual(learner._learner.get_decay(), self.float_1)
        learner.decay = self.float_2
        self.assertEqual(learner.decay, self.float_2)
        self.assertEqual(learner._learner.get_decay(), self.float_2)

        learner = HawkesADM4(1, rho=self.float_1)
        self.assertEqual(learner.rho, self.float_1)
        self.assertEqual(learner._learner.get_rho(), self.float_1)
        learner.rho = self.float_2
        self.assertEqual(learner.rho, self.float_2)
        self.assertEqual(learner._learner.get_rho(), self.float_2)


if __name__ == '__main__':
    unittest.main()
