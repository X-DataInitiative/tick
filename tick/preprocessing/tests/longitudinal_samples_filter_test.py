# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
import unittest
from tick.preprocessing import LongitudinalSamplesFilter


class Test(unittest.TestCase):
    def setUp(self):
        self.features = [
            np.array([[0, 1, 0], [0, 0, 0], [0, 1, 1]], dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
            np.zeros((3, 3), dtype="float64"),
            np.array([[1, 1, 1], [0, 0, 1], [1, 1, 0]], dtype="float64"),
        ]
        self.sparse_features = [csr_matrix(f) for f in self.features]

        self.labels = [
            np.array([0, 0, 1], dtype="float64"),
            np.array([1, 0, 0], dtype="float64"),
            np.array([0, 1, 0], dtype="float64"),
            np.zeros((3,), dtype="float64")
        ]

        self.censoring = np.array([2, 3, 3, 1], dtype="uint64")

        self.expected_output = (self.features[0:2], self.labels[0:2],
                                self.censoring[0:2])

    def test_dense_fitlering(self):
        output = LongitudinalSamplesFilter()\
            .fit_transform(self.features, self.labels, self.censoring)
        [
            np.testing.assert_equal(out, self.expected_output[i])
            for i, out in enumerate(output)
        ]

    def test_sparse_filtering(self):
        output = LongitudinalSamplesFilter()\
            .fit_transform(self.sparse_features, self.labels, self.censoring)
        np.testing.assert_equal([out.todense() for out in output[0]],
                                self.expected_output[0])
        [
            np.testing.assert_equal(out, self.expected_output[i + 1])
            for i, out in enumerate(output[1:])
        ]


if __name__ == "__main__":
    unittest.main()
