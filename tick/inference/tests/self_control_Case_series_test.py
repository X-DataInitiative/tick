import unittest

import itertools
from itertools import product
import numpy as np
from tick.inference.tests.inference import InferenceTest
from tick.simulation import SimuSCCS
from tick.inference import LearnerSCCS


class Test(InferenceTest):
    def setUp(self):
        # Create some data here
        sim = SimuSCCS(n_samples=300, n_intervals=10, n_features=3, n_lags=4,
                       verbose=False, seed=42)
        self.n_lags = 4
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
        lrn = LearnerSCCS(n_lags=self.n_lags)
        X, y, c = lrn._preprocess(self.features, self.labels, self.censoring)
        pass

    def test_LearnerSCCS_fit(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        pass

    def test_LearnerSCCS_refit(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        coeffs_refit = lrn._refit(coeffs, self.features, self.labels, self.censoring)
        pass

    def test_LearnerSCCS_bootstrap_CI(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        coeffs_refit = lrn.bootstrap_CI(coeffs, self.features,
                                        self.censoring, 10, 0.95,
                                        random_state=42)
        # TODO: no verbose here
        pass

    def test_LearnerSCCS_score(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit(self.features, self.labels, self.censoring)
        lrn.score()
        lrn.score(self.features, self.labels, self.censoring)
        pass

    def test_LearnerSCCS_fit_KFold_CV(self):
        lrn = LearnerSCCS(n_lags=self.n_lags)
        coeffs = lrn.fit_KFold_CV(self.features, self.labels, self.censoring,
                                  strength_TV_list=[1e-3, 1e-4], stratified=False)
        pass

    def test_LearnerSCCS_warm_start(self):
        pass

    def test_LearnerSCCS_settings(self):
        pass