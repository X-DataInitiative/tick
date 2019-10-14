# License: BSD 3 clause

import os
import pickle
import unittest

import numpy as np

from tick.base.inference import InferenceTest

skip = True
try:
    import tensorflow as tf
    if int(tf.__version__[0]) == 1: # test is disabled until v2 is working
        skip = False
except ImportError:
    print("tensorflow not found, skipping HawkesCumulantMatching test")

if not skip:
    from tick.hawkes import HawkesCumulantMatching

    class Test(InferenceTest):
        def setUp(self):
            self.dim = 2
            np.random.seed(320982)

        @staticmethod
        def get_train_data(decay):
            saved_train_data_path = os.path.join(
                os.path.dirname(__file__),
                'hawkes_cumulant_matching_test-train_data.pkl')

            with open(saved_train_data_path, 'rb') as f:
                train_data = pickle.load(f)

            baseline = train_data[decay]['baseline']
            adjacency = train_data[decay]['adjacency']
            timestamps = train_data[decay]['timestamps']

            return timestamps, baseline, adjacency

        def test_hawkes_cumulants(self):
            """...Test that estimated cumulants are coorect
            """
            timestamps, baseline, adjacency = Test.get_train_data(decay=3.)

            expected_L = [2.149652, 2.799746, 4.463995]

            expected_C = [[15.685827, 16.980316,
                           30.232248], [16.980316, 23.765304, 36.597161],
                          [30.232248, 36.597161, 66.271089]]

            expected_K = [[49.179092, -959.246309, -563.529052],
                          [-353.706952, -1888.600201, -1839.608349],
                          [-208.913969, -2103.952235, -150.937999]]

            learner = HawkesCumulantMatching(100.)
            learner._set_data(timestamps)
            self.assertFalse(learner._cumulant_computer.cumulants_ready)
            learner.compute_cumulants()
            self.assertTrue(learner._cumulant_computer.cumulants_ready)

            np.testing.assert_array_almost_equal(learner.mean_intensity,
                                                 expected_L)
            np.testing.assert_array_almost_equal(learner.covariance,
                                                 expected_C)
            np.testing.assert_array_almost_equal(learner.skewness, expected_K)

            self.assertAlmostEqual(learner.approximate_optimal_cs_ratio(),
                                   0.999197628503)

            learner._set_data(timestamps)
            self.assertTrue(learner._cumulant_computer.cumulants_ready)

        def test_hawkes_cumulants_solve(self):
            """...Test that hawkes cumulant reached expected value
            """
            timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
            learner = HawkesCumulantMatching(100., cs_ratio=0.9, max_iter=300,
                                             print_every=30, step=1e-2,
                                             solver='adam', C=1e-3, tol=1e-5)
            learner.fit(timestamps)

            expected_R_pred = [[0.423305, -0.559607,
                                -0.307212], [-0.30411, 0.27066, -0.347162],
                               [0.484648, 0.331057, 1.591584]]

            np.testing.assert_array_almost_equal(learner.solution,
                                                 expected_R_pred)

            expected_baseline = [36.808583, 32.304106, -15.123118]

            np.testing.assert_array_almost_equal(learner.baseline,
                                                 expected_baseline)

            expected_adjacency = [[-3.34742247, -6.28527387, -2.21012092],
                                  [-2.51556256, -5.55341413, -1.91501755],
                                  [1.84706793, 3.2770494, 1.44302449]]

            np.testing.assert_array_almost_equal(learner.adjacency,
                                                 expected_adjacency)

            np.testing.assert_array_almost_equal(
                learner.objective(learner.adjacency), 149029.4540306161)

            np.testing.assert_array_almost_equal(
                learner.objective(R=learner.solution), 149029.4540306161)

            # Ensure learner can be fit again
            timestamps_2, baseline, adjacency = Test.get_train_data(decay=2.)
            learner.step = 1e-1
            learner.penalty = 'l2'
            learner.fit(timestamps_2)

            expected_adjacency_2 = [[-0.021966, -0.178811, -0.107636],
                                    [0.775206, 0.384494,
                                     0.613925], [0.800584, 0.581281, 0.60177]]

            np.testing.assert_array_almost_equal(learner.adjacency,
                                                 expected_adjacency_2)

            learner_2 = HawkesCumulantMatching(
                100., cs_ratio=0.9, max_iter=299, print_every=30, step=1e-1,
                solver='adam', penalty='l2', C=1e-3, tol=1e-5)
            learner_2.fit(timestamps_2)

            np.testing.assert_array_almost_equal(learner.adjacency,
                                                 expected_adjacency_2)

            # Check cumulants are not computed again
            learner_2.step = 1e-2
            learner_2.fit(timestamps_2)

        def test_hawkes_cumulants_unfit(self):
            """...Test that HawkesCumulantMatching raises an error if no data is
            given
            """
            learner = HawkesCumulantMatching(100., cs_ratio=0.9, max_iter=299,
                                             print_every=30, step=1e-2,
                                             solver='adam')

            msg = '^Cannot compute cumulants if no realization has been provided$'
            with self.assertRaisesRegex(RuntimeError, msg):
                learner.compute_cumulants()

        def test_hawkes_cumulants_solve_l1(self):
            """...Test that hawkes cumulant reached expected value with l1
            penalization
            """
            timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
            learner = HawkesCumulantMatching(
                100., cs_ratio=0.9, max_iter=300, print_every=30, step=1e-2,
                solver='adam', penalty='l1', C=1, tol=1e-5)
            learner.fit(timestamps)

            expected_R_pred = [[0.434197, -0.552021,
                                -0.308883], [-0.299366, 0.272764, -0.347764],
                               [0.48448, 0.331059, 1.591587]]

            np.testing.assert_array_almost_equal(learner.solution,
                                                 expected_R_pred)

            expected_baseline = [32.788801, 29.324684, -13.275885]

            np.testing.assert_array_almost_equal(learner.baseline,
                                                 expected_baseline)

            expected_adjacency = [[-2.925945, -5.54899, -1.97438],
                                  [-2.201373, -5.009153,
                                   -1.740234], [1.652958, 2.939054, 1.334677]]

            np.testing.assert_array_almost_equal(learner.adjacency,
                                                 expected_adjacency)

            np.testing.assert_array_almost_equal(
                learner.objective(learner.adjacency), 149061.5590630687)

            np.testing.assert_array_almost_equal(
                learner.objective(R=learner.solution), 149061.5590630687)

        def test_hawkes_cumulants_solve_l2(self):
            """...Test that hawkes cumulant reached expected value with l2
            penalization
            """
            timestamps, baseline, adjacency = Test.get_train_data(decay=3.)
            learner = HawkesCumulantMatching(
                100., cs_ratio=0.9, max_iter=300, print_every=30, step=1e-2,
                solver='adam', penalty='l2', C=0.1, tol=1e-5)
            learner.fit(timestamps)

            expected_R_pred = [[0.516135, -0.484529,
                                -0.323191], [-0.265853, 0.291741, -0.35285],
                               [0.482819, 0.331344, 1.591535]]

            np.testing.assert_array_almost_equal(learner.solution,
                                                 expected_R_pred)

            expected_baseline = [17.066997, 17.79795, -6.07811]

            np.testing.assert_array_almost_equal(learner.baseline,
                                                 expected_baseline)

            expected_adjacency = [[-1.310854, -2.640152, -1.054596],
                                  [-1.004887, -2.886297,
                                   -1.065671], [0.910245, 1.610029, 0.913469]]

            np.testing.assert_array_almost_equal(learner.adjacency,
                                                 expected_adjacency)

            np.testing.assert_array_almost_equal(
                learner.objective(learner.adjacency), 149232.94041039888)

            np.testing.assert_array_almost_equal(
                learner.objective(R=learner.solution), 149232.94041039888)


if __name__ == "__main__":
    if not skip:
        unittest.main()
