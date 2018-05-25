# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing import LongitudinalFeaturesProduct
from tick.preprocessing.build.preprocessing \
    import SparseLongitudinalFeaturesProduct
import pickle


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

        self.n_intervals, self.n_features = self.finite_exposures[0].shape

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
        pp = LongitudinalFeaturesProduct("finite", n_jobs=1)

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
        feat_prod, _, _ = LongitudinalFeaturesProduct("infinite", n_jobs=1)\
            .fit_transform(self.sparse_infinite_exposures)
        feat_prod = [f.toarray() for f in feat_prod]
        np.testing.assert_equal(feat_prod, expected_output)

    def test_parallelization(self):
        for exp_type in ['finite', 'infinite']:
            feat_prod, _, _ = LongitudinalFeaturesProduct(exp_type, n_jobs=1)\
                .fit_transform(self.sparse_infinite_exposures)
            p_feat_prod, _, _ = LongitudinalFeaturesProduct(exp_type,
                                                            n_jobs=-1)\
                .fit_transform(self.sparse_infinite_exposures)
            feat_prod = [f.toarray() for f in feat_prod]
            p_feat_prod = [f.toarray() for f in p_feat_prod]
            np.testing.assert_equal(feat_prod, p_feat_prod)

        exp_type = 'finite'
        feat_prod, _, _ = LongitudinalFeaturesProduct(exp_type, n_jobs=1) \
            .fit_transform(self.finite_exposures)
        p_feat_prod, _, _ = LongitudinalFeaturesProduct(exp_type, n_jobs=-1) \
            .fit_transform(self.finite_exposures)
        np.testing.assert_equal(feat_prod, p_feat_prod)

    def test_serialization(self):
        python = LongitudinalFeaturesProduct()
        cpp = SparseLongitudinalFeaturesProduct(self.n_features)
        pickle.loads(pickle.dumps(python))
        pickle.loads(pickle.dumps(cpp))
        # Cannot check equality as CPP underlying objects will not be created in
        # the same memory slot


if __name__ == "__main__":
    unittest.main()
