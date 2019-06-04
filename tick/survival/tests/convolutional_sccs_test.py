import unittest
import numpy as np
from tick.survival import SimuSCCS, ConvSCCS
from scipy.sparse import csr_matrix
from tick.survival.sccs import *

class Test(unittest.TestCase):
    def setUp(self):
        self.n_lags = np.repeat(1, 2).astype('uint64')
        self.seed = 42
        self.coeffs = [
            np.log(np.array([2.1, 2.5])),
            np.log(np.array([.8, .5]))
        ]
        self.n_features = len(self.n_lags)
        self.n_correlations = 2
        # Create data
        sim = SimuSCCS(n_cases=500, n_intervals=10, n_features=self.n_features,
                       n_lags=self.n_lags, verbose=False, seed=self.seed,
                       coeffs=self.coeffs, n_correlations=self.n_correlations)
        _, self.features, self.labels, self.censoring, self.coeffs =\
            sim.simulate()

    def test_LearnerSCCS_coefficient_groups(self):
        n_lags = np.array([4, 0, 3, 4], dtype='uint64')
        n_features = len(n_lags)
        n_coeffs = (n_lags + 1).sum()
        coeffs = np.ones((n_coeffs,))
        # 1st feature
        coeffs[1:3] = 2
        # 2nd feature
        coeffs[5] = 1
        # 3rd feature
        coeffs[6:8] = 0
        # 4th feature
        coeffs[10:] = np.array([1, 2, 3, 4, 4])
        expected_equality_groups = [(1, 3), (3, 5), (6, 8), (8, 10), (13, 15)]
        lrn = ConvSCCS(n_lags=n_lags, penalized_features=np.arange(4))
        lrn._set("n_features", n_features)
        equality_groups = lrn._detect_support(coeffs)
        self.assertEqual(expected_equality_groups, equality_groups)

    def test_LearnerSCCS_preprocess(self):
        features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64")
        ]
        sparse_features = [csr_matrix(f, shape=(3, 3)) for f in features]
        labels = [
            np.array([0, 0, 1], dtype="uint64"),
            np.array([0, 1, 0], dtype="uint64"),
            np.array([0, 0, 0], dtype="uint64")
        ]
        censoring = np.array([2, 3, 3], dtype="uint64")
        n_lags = np.array([1, 1, 0], dtype="uint64")

        expected_features = [
            np.array([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]),
            np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [1, 0, 0, 0, 0]])
        ]
        expected_labels = [
            np.array([0, 0, 1], dtype="uint64"),
            np.array([0, 1, 0], dtype="uint64"),
        ]
        expected_censoring = np.array([2, 3], dtype="uint64")

        lrn = ConvSCCS(n_lags=n_lags, penalized_features=[])
        X, y, c = lrn._prefit(sparse_features, labels, censoring)
        [
            np.testing.assert_array_equal(f.toarray(), expected_features[i])
            for i, f in enumerate(X)
        ]
        [
            np.testing.assert_array_equal(l, expected_labels[i])
            for i, l in enumerate(y)
        ]
        np.testing.assert_array_equal(c, expected_censoring)

    def test_LearnerSCCS_fit(self):
        seed = 42
        n_lags = np.repeat(2, 2).astype('uint64')
        sim = SimuSCCS(n_cases=800, n_intervals=10, n_features=2,
                       n_lags=n_lags, verbose=False, seed=seed,
                       exposure_type='multiple_exposures')
        features, _, labels, censoring, coeffs = sim.simulate()
        lrn = ConvSCCS(n_lags=n_lags, penalized_features=[], tol=0,
                       max_iter=10, random_state=seed)
        estimated_coeffs, _ = lrn.fit(features, labels, censoring)
        np.testing.assert_almost_equal(
            np.hstack(estimated_coeffs), np.hstack(coeffs), decimal=1)

    def test_LearnerSCCS_confidence_intervals(self):
        def fit(clazz):
            self.setUp()
            lrn = clazz(n_lags=self.n_lags, penalized_features=[])
            coeffs, _ = lrn.fit(self.features, self.labels, self.censoring)
            p_features, p_labels, p_censoring = lrn._preprocess_data(
                self.features, self.labels, self.censoring)
            confidence_intervals = lrn._bootstrap(p_features,
                                                  p_labels, p_censoring,
                                                  np.hstack(coeffs), 5, .90)
            for i, c in enumerate(coeffs):
                self.assertTrue(
                    np.all(confidence_intervals.lower_bound[i] <= c),
                    "lower bound of the confidence interval\
                                       should be <= coeffs at index %i" % i)
                self.assertTrue(
                    np.all(c <= confidence_intervals.upper_bound[i]),
                    "upper bound of the confidence interval\
                                       should be >= coeffs at index %i" % i)
            # Same with 0 lags
            n_lags = np.zeros_like(self.n_lags, dtype='uint64')
            lrn = clazz(n_lags=n_lags, penalized_features=[])
            coeffs, _ = lrn.fit(self.features, self.labels, self.censoring)
            p_features, p_labels, p_censoring = lrn._preprocess_data(
                self.features, self.labels, self.censoring)
            confidence_intervals = lrn._bootstrap(p_features,
                                                  p_labels, p_censoring,
                                                  np.hstack(coeffs), 5, .90)
            for i, c in enumerate(coeffs):
                self.assertTrue(
                    np.all(confidence_intervals.lower_bound[i] <= c),
                    "lower bound of the confidence interval\
                                       should be <= coeffs at index %i" % i)
                self.assertTrue(
                    np.all(c <= confidence_intervals.upper_bound[i]),
                    "upper bound of the confidence interval\
                                       should be >= coeffs at index %i" % i)

        fit(ConvSCCS)
        fit(BatchConvSCCS)
        fit(StreamConvSCCS)

    def test_LearnerSCCS_score(self):
        lrn = ConvSCCS(n_lags=self.n_lags, penalized_features=[],
                       random_state=self.seed)
        lrn.fit(self.features, self.labels, self.censoring)
        self.assertEqual(lrn.score(),
                         lrn.score(self.features, self.labels, self.censoring))

    def test_LearnerSCCS_fit_KFold_CV(self):
        lrn_scores = []
        def fit(lrn):
            self.setUp()
            lrn.fit(self.features, self.labels, self.censoring)
            score = lrn.score()
            tv_range = (-5, -1)
            groupl1_range = (-5, -1)
            lrn.fit_kfold_cv(self.features, self.labels, self.censoring,
                             C_tv_range=tv_range, C_group_l1_range=groupl1_range,
                             n_cv_iter=4)
            lrn_scores.append(lrn.score())
            self.assertTrue(lrn_scores[-1] <= score)

        fit(ConvSCCS(n_lags=self.n_lags, penalized_features=np.arange(
            self.n_features), random_state=self.seed, C_tv=1e-1,
                       C_group_l1=1e-1))
        fit(BatchConvSCCS(n_lags=self.n_lags, penalized_features=np.arange(
            self.n_features), random_state=self.seed, C_tv=1e-1,
                       C_group_l1=1e-1))
        fit(StreamConvSCCS(n_lags=self.n_lags, penalized_features=np.arange(
            self.n_features), random_state=self.seed, C_tv=1e-1,
                       C_group_l1=1e-1))

if __name__ == "__main__":
    unittest.main()
