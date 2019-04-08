# License: BSD 3 clause

import os
import unittest
import uuid
import gc

from scipy import sparse
import numpy as np

from tick.array.serialize import serialize_array, load_array


class Test(object):
    def __init__(self, *args, dtype="float64", **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dtype = dtype

    def setUp(self):
        # ensure all tests use their own file
        self.array_file = 'tmp_array_file_{}.cereal'.format(uuid.uuid4())

    def tearDown(self):
        if os.path.exists(self.array_file):
            os.remove(self.array_file)
        else:
            raise FileNotFoundError('array has not been stored to file')

    def test_serialize_1d_array(self):
        """...Test serialization of 1d dense array is done as expected
        """
        array = np.random.rand(100).astype(self.dtype)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, dtype=self.dtype)
        np.testing.assert_array_almost_equal(array, serialized_array)

    def test_serialize_2d_array(self):
        """...Test serialization of 2d dense array is done as expected
        """
        array = np.random.rand(10, 10).astype(self.dtype)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, array_dim=2,
                                      dtype=self.dtype)
        np.testing.assert_array_almost_equal(array, serialized_array)

    def test_serialize_sparse_2d_array(self):
        """...Test serialization of 2d sparse array is done as expected
        """
        array = sparse.rand(10, 10, density=0.3,
                            format='csr').astype(self.dtype)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, array_dim=2,
                                      array_type='sparse', dtype=self.dtype)
        np.testing.assert_array_almost_equal(array.toarray(),
                                             serialized_array.toarray())

        # python 3.5 has show to required this - investigate typemappers
        gc.collect()

    def test_serialize_column_major_2d_array(self):
        """...Test serialization of 2d dense array is done as expected
        """
        row_array = np.arange(80).reshape(10, 8).astype(self.dtype)

        col_array = np.asfortranarray(row_array)
        serialize_array(col_array, self.array_file)
        serialized_col_array = load_array(self.array_file, array_dim=2,
                                              dtype=self.dtype,
                                              major="col")
        np.testing.assert_array_almost_equal(col_array, row_array)
        np.testing.assert_array_almost_equal(col_array, np.asfortranarray(row_array))
        np.testing.assert_array_almost_equal(col_array.flatten('K'), serialized_col_array.flatten('K'))
        np.testing.assert_array_almost_equal(col_array, serialized_col_array)

    def test_serialize_column_major_sparse_2d_array(self):
        """...Test serialization of 2d sparse array is done as expected
        """
        array = sparse.rand(10, 10, density=0.3,
                            format='csc').astype(self.dtype)
        serialize_array(array, self.array_file)

        serialized_array = load_array(self.array_file, array_dim=2,
                                      array_type='sparse', dtype=self.dtype,
                                      major="col")
        np.testing.assert_array_almost_equal(array.toarray(),
                                             serialized_array.toarray())

        # python 3.5 has show to required this - investigate typemappers
        gc.collect()


class TestFloat32(Test, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        Test.__init__(self, *args, dtype="float32", **kwargs)


class TestFloat64(Test, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        Test.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
