# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing import LongitudinalFeaturesLagger


class Test(unittest.TestCase):
    def setUp(self):
        self.features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64")
        ]
        self.sparse_features = [csr_matrix(f) for f in self.features]

        self.censoring = np.array([2, 3], dtype="uint64")

        self.expected_output = [
            np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0.]]),
            np.array([[1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 1],
                      [1, 0, 1, 0, 1, 0, 1.]])
        ]

        self.n_lags = np.array([1, 2, 1], dtype="uint64")

    def test_dense_pre_convolution(self):
        feat_prod, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags)\
            .fit_transform(self.features, censoring=self.censoring)
        np.testing.assert_equal(feat_prod, self.expected_output)

    def test_sparse_pre_convolution(self):
        feat_prod, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags)\
            .fit_transform(self.sparse_features, censoring=self.censoring)
        feat_prod = [f.todense() for f in feat_prod]
        np.testing.assert_equal(feat_prod, self.expected_output)


if __name__ == "__main__":
    unittest.main()
