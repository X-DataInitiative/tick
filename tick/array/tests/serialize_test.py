# License: BSD 3 clause

import os
import unittest

import numpy as np
import scipy

from tick.array.serialize import serialize_array, load_array
from tick.solver.tests.solver import TestSolver


class Test(TestSolver):
    def setUp(self):
        self.array_file = 'tmp_array_file.cereal'

    def tearDown(self):
        if os.path.exists(self.array_file):
            os.remove(self.array_file)

    def test_serialize_1d_array(self):
        """...Test serialization of 1d dense array is done as expected
        """
        array = np.random.rand(100)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file)
        np.testing.assert_array_almost_equal(array, serialized_array)

    def test_serialize_2d_array(self):
        """...Test serialization of 2d dense array is done as expected
        """
        array = np.random.rand(10, 10)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, array_dim=2)
        np.testing.assert_array_almost_equal(array, serialized_array)

    def test_serialize_sparse_2d_array(self):
        """...Test serialization of 2d dense array is done as expected
        """
        array = scipy.sparse.rand(10, 10, density=0.3, format='csr')
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, array_dim=2,
                                      array_type='sparse')
        np.testing.assert_array_almost_equal(array.toarray(),
                                             serialized_array.toarray())


if __name__ == '__main__':
    unittest.main()
