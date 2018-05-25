# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing import LongitudinalFeaturesLagger
from tick.preprocessing.build.preprocessing import LongitudinalFeaturesLagger\
    as _LongitudinalFeaturesLagger
import pickle


class Test(unittest.TestCase):
    def setUp(self):
        self.features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64")
        ]
        self.sparse_features = [csr_matrix(f) for f in self.features]
        self.n_intervals, self.n_features = self.features[0].shape

        self.censoring = np.array([2, 3], dtype="uint64")

        self.expected_output = [
            np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0.]]),
            np.array([[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1],
                      [1, 0, 1, 0, 1, 0, 1.]])
        ]

        self.n_lags = np.array([1, 2, 1], dtype="uint64")

    def test_dense_pre_convolution(self):
        lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                       n_jobs=1)\
            .fit_transform(self.features, censoring=self.censoring)
        np.testing.assert_equal(lagged_feat, self.expected_output)

    def test_sparse_pre_convolution(self):
        lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                       n_jobs=1)\
            .fit_transform(self.sparse_features, censoring=self.censoring)
        lagged_feat = [f.todense() for f in lagged_feat]
        np.testing.assert_equal(lagged_feat, self.expected_output)

    def test_parallelization_sparse(self):
        lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                       n_jobs=1) \
            .fit_transform(self.sparse_features, censoring=self.censoring)
        p_lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                         n_jobs=-1)\
            .fit_transform(self.sparse_features, censoring=self.censoring)
        lagged_feat = [f.toarray() for f in lagged_feat]
        p_lagged_feat = [f.toarray() for f in p_lagged_feat]
        np.testing.assert_equal(lagged_feat, p_lagged_feat)

    def test_parallelization_dense(self):
        lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                       n_jobs=1) \
            .fit_transform(self.features, censoring=self.censoring)
        p_lagged_feat, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags,
                                                         n_jobs=-1)\
            .fit_transform(self.features, censoring=self.censoring)
        np.testing.assert_equal(lagged_feat, p_lagged_feat)

    def test_serialization(self):
        python = LongitudinalFeaturesLagger(n_lags=self.n_lags, n_jobs=1)
        cpp = _LongitudinalFeaturesLagger(self.n_intervals, self.n_lags)
        pickle.loads(pickle.dumps(python))
        pickle.loads(pickle.dumps(cpp))
        # Cannot check equality as CPP underlying objects will not be created in
        # the same memory slot


if __name__ == "__main__":
    unittest.main()
