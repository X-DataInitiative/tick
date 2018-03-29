# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing import LongitudinalFeaturesProduct


class Test(unittest.TestCase):
    def setUp(self):
        self.finite_exposures = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64")
        ]
        self.sparse_finite_exposures = [
            csr_matrix(f) for f in self.finite_exposures
        ]

        self.infinite_exposures = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype="float64"),
            np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype="float64")
        ]
        self.sparse_infinite_exposures = [
            csr_matrix(f) for f in self.infinite_exposures
        ]

    def test_finite_features_product(self):
        expected_output = \
            [np.array([[0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 1, 1, 0, 0, 1],
                       ], dtype="float64"),
             np.array([[1, 1, 1, 1, 1, 1],
                       [0, 0, 1, 0, 0, 0],
                       [1, 1, 0, 1, 0, 0],
                       ], dtype="float64")
             ]
        pp = LongitudinalFeaturesProduct("finite")

        feat_prod, _, _ = pp.fit_transform(self.finite_exposures)
        np.testing.assert_equal(feat_prod, expected_output)

        feat_prod, _, _ = pp.fit_transform(self.sparse_finite_exposures)
        feat_prod = [f.toarray() for f in feat_prod]
        np.testing.assert_equal(feat_prod, expected_output)

    def test_sparse_infinite_features_product(self):
        expected_output = \
            [np.array([[0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 1],
                       ], dtype="float64"),
             np.array([[1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 1, 1],
                       [0, 0, 0, 0, 0, 0],
                       ], dtype="float64")
             ]
        sparse_feat = [csr_matrix(f) for f in self.infinite_exposures]
        feat_prod, _, _ = LongitudinalFeaturesProduct("infinite")\
            .fit_transform(sparse_feat)
        feat_prod = [f.toarray() for f in feat_prod]
        np.testing.assert_equal(feat_prod, expected_output)


if __name__ == "__main__":
    unittest.main()
