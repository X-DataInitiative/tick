import unittest

import itertools
from itertools import product
import numpy as np
from tick.inference.tests.inference import InferenceTest
from tick.simulation import SimuSCCS
from tick.inference import LearnerSCCS
from tick.optim.model import ModelSCCS
from tick.optim.solver import SVRG
from tick.optim.prox import ProxZero


class Test(InferenceTest):
    # TODO: remove verbose everywhere
    def setUp(self):
        # Create data
        sim = SimuSCCS(n_samples=5000, n_intervals=50, n_features=2, n_lags=3,
                       verbose=False, seed=42)
        self.n_lags = 3
        self.features, self.labels, self.censoring, self.coeffs = sim.simulate()

    def test_LearnerSCCS_coefficient_groups(self):
        coeffs = np.ones((3, 4))
        coeffs[0, 1:3] = 2
        coeffs[1, 2:] = 4
        coeffs[2, 0:2] = 0
        expected_equality_groups = [(0, 1), (1, 3), (3, 4),
                                    (4, 6), (6, 8),
                                    (8, 10), (10, 12)]
        expected_tv_groups = [(0, 4), (4, 8), (8, 12)]
        lrn = LearnerSCCS(n_lags=3)
        lrn._set("n_features", 3)
        equality_groups = lrn._coefficient_groups("Equality", coeffs)
        tv_groups = lrn._coefficient_groups("TV", coeffs)
        l1_tv_groups = lrn._coefficient_groups("L1-TV", coeffs)
        self.assertEqual(expected_equality_groups, equality_groups)
        self.assertEqual(expected_tv_groups, tv_groups)
        self.assertEqual(expected_tv_groups, l1_tv_groups)

    def test_LearnerSCCS_preprocess(self):
        # Just check that the preprocessing is running quickly
        lrn = LearnerSCCS(n_lags=self.n_lags)
        X, y, c = lrn._preprocess(self.features, self.labels, self.censoring)
        # TODO: Check on small dummy data that preprocessing is working
        pass

    def test_LearnerSCCS_fit(self):
        # TODO: correct this test
        lrn = LearnerSCCS(n_lags=self.n_lags, penalty="None", tol=0,
                          max_iter=20)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)

        p_features, p_labels, p_censoring = lrn._preprocess(self.features,
                                                            self.labels,
                                                            self.censoring)
        model = ModelSCCS(n_intervals=lrn.n_intervals,
                          n_lags=self.n_lags).fit(p_features, p_labels,
                                                  p_censoring)
        solver = SVRG(max_iter=15, verbose=False)
        solver.set_model(model).set_prox(ProxZero())
        coeffs_svrg = solver.solve(step=1 / model.get_lip_max())
        np.testing.assert_almost_equal(coeffs, coeffs_svrg, decimal=1)
        np.testing.assert_almost_equal(coeffs, self.coeffs, decimal=1)

    def test_LearnerSCCS_bootstrap_CI(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        p_features, p_labels, p_censoring = lrn._preprocess(self.features,
                                                            self.labels,
                                                            self.censoring)
        coeffs, lb, ub = lrn._bootstrap(p_features, p_labels, p_censoring,
                                        100, 0.05)
        self.assertTrue(all(lb <= coeffs),
                        "lower bound of the confidence interval\
                               should be <= coeffs")
        self.assertTrue(all(coeffs <= ub),
                        "upper bound of the confidence interval\
                               should be >= coeffs")

    def test_LearnerSCCS_score(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        lrn.score()
        lrn.score(self.features, self.labels, self.censoring)
        # TODO: compare with model score for this data
        pass

    def test_LearnerSCCS_fit_KFold_CV(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit_KFold_CV(self.features, self.labels, self.censoring,
                                  strength_TV_list=[1e-3, 1e-4], stratified=False)
        # TODO: check that score <= score when no penalization
        pass

    def test_LearnerSCCS_settings(self):
        # TODO: test all the settings here
        pass