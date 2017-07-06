# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing.longitudinal_features_product import LongitudinalFeaturesProduct


class Test(unittest.TestCase):

    def setUp(self):
        self.short_exposures = [np.array([[0, 1, 0],
                                          [0, 0, 0],
                                          [0, 1, 1]], dtype="float64"),
                                np.array([[1, 1, 1],
                                          [0, 0, 1],
                                          [1, 1, 0]], dtype="float64")
                                ]
        self.sparse_short_exposures = [csr_matrix(f)
                                       for f in self.short_exposures]

        self.infinite_exposures = [np.array([[0, 1, 0],
                                             [0, 0, 0],
                                             [0, 0, 1]], dtype="float64"),
                                   np.array([[1, 1, 0],
                                             [0, 0, 1],
                                             [0, 0, 0]], dtype="float64")
                                   ]
        self.sparse_infinite_exposures = [csr_matrix(f)
                                          for f in self.infinite_exposures]


    def test_short_features_product(self):
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
        pp = LongitudinalFeaturesProduct("short")

        feat_prod = pp.fit_transform(self.short_exposures)
        np.testing.assert_equal(feat_prod, expected_output)

        feat_prod = pp.fit_transform(self.sparse_short_exposures)
        print(feat_prod.__class__)
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
        feat_prod = LongitudinalFeaturesProduct("infinite")\
            .fit_transform(sparse_feat)
        feat_prod = [f.toarray() for f in feat_prod]
        np.testing.assert_equal(feat_prod, expected_output)


if __name__ == "__main__":
    unittest.main()