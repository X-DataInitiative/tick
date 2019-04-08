# License: BSD 3 clause

# -*- coding: utf8 -*-

import unittest
import numpy as np
from numpy.linalg import norm
from tick.array_test.build import array_test as test
from scipy.sparse import csr_matrix
import itertools


class Test(unittest.TestCase):
    def setUp(self):
        self.python_array_1 = np.array([1, 2.0, -5, 0, 4, 1, -8, np.pi, 0])
        self.python_array_2 = np.array([17, 0.25, -5.9, 0, 4, 1, 3, 2, 22])

        self.python_array_2d = np.array([[1, 2.1, 5], [0, 4, 1]])

        self.python_sparse_array_1 = csr_matrix((np.array([1.5, 2, 3, 1]),
                                                 np.array([3, 5, 7, 8]),
                                                 np.array([0, 4])))

        self.python_sparse_array_2 = csr_matrix((np.array(
            [1.5, -3, np.sqrt(2), 0.9, 4]), np.array([1, 2, 3, 6, 8]),
                                                 np.array([0, 5])))

        self.python_sparse_array_3 = csr_matrix(
            (np.array([-2.]), np.array([8]), np.array([0, 1])))

        self.python_sparse_array_2d = csr_matrix((np.array([1.5, 2, 3, 1]),
                                                  np.array([3, 5, 7, 4]),
                                                  np.array([0, 3, 4])))

        self.array_types = [
            'BaseArrayDouble', 'ArrayDouble', 'SparseArrayDouble',
            'BaseArrayDouble2d', 'ArrayDouble2d', 'SparseArrayDouble2d',
            'SBaseArrayDoublePtr', 'SArrayDoublePtr', 'VArrayDoublePtr',
            'SSparseArrayDoublePtr', 'SBaseArrayDouble2dPtr',
            'SArrayDouble2dPtr', 'SSparseArrayDouble2dPtr'
        ]

    def _test_arrays(self, array_type):
        test_arrays = []
        if Test.is_1d(array_type):
            if Test.is_sparse(array_type) or Test.is_base(array_type):
                test_arrays.append(self.python_sparse_array_1.copy())
                test_arrays.append(self.python_sparse_array_2.copy())
                test_arrays.append(self.python_sparse_array_3.copy())
            if Test.is_dense(array_type) or Test.is_base(array_type):
                test_arrays.append(self.python_array_1.copy())
                test_arrays.append(self.python_array_2.copy())
        else:
            if Test.is_sparse(array_type) or Test.is_base(array_type):
                test_arrays.append(self.python_sparse_array_2d.copy())
            if Test.is_dense(array_type) or Test.is_base(array_type):
                test_arrays.append(self.python_array_2d.copy())
        return test_arrays

    @staticmethod
    def is_sparse(array_type):
        return 'Sparse' in array_type

    @staticmethod
    def is_base(array_type):
        return 'Base' in array_type

    @staticmethod
    def is_dense(array_type):
        return not Test.is_sparse(array_type) and not Test.is_base(array_type)

    @staticmethod
    def is_array_ptr(array_type):
        return 'Ptr' in array_type

    @staticmethod
    def is_1d(array_type):
        return '2d' not in array_type

    @staticmethod
    def compare_arrays(array1, array2):
        if hasattr(array1, "toarray"):
            array1 = array1.toarray()
        if hasattr(array2, "toarray"):
            array2 = array2.toarray()
        np.testing.assert_almost_equal(array1, array2)

    @staticmethod
    def cast_1d_array_to_1d_dense(array):
        if hasattr(array, "toarray"):
            # If the array was a sparse array we reshape it into
            # a 1 dimensional array
            array = array.toarray()
            array = array.reshape(np.prod(array.shape))
        elif hasattr(array, "A1"):
            array = array.A1
        return array

    @staticmethod
    def cast_array_to_dense(array):
        if hasattr(array, "toarray"):
            # If the array was a sparse array we reshape it into
            # a 1 dimensional array
            array = array.toarray()
        return array

    def test_init_to_zero(self):
        """...Test all type of arrays can be init to zero"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_init_to_zero_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                test_func(test_array)
                if (isinstance(test_array, np.ndarray)):
                    Test.compare_arrays(test_array, np.zeros_like(test_array))
                else:
                    Test.compare_arrays(test_array.data,
                                        np.zeros_like(test_array.data))

    def test_copy(self):
        """...Test behavior of copy constructor and assignment on arrays"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_copy_%s" % array_type)
            original_arrays = self._test_arrays(array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array, original_array in zip(test_arrays,
                                                  original_arrays):
                test_func(test_array)
                # If it is a shared ptr it should have set the pointed array
                # to 0
                if Test.is_array_ptr(array_type):
                    Test.compare_arrays(test_array, np.zeros_like(test_array))
                # If it is not a pointer, data has been copy and array passed
                # as argument should not have been modified
                else:
                    Test.compare_arrays(test_array, original_array)

    def test_move(self):
        """...Test behavior of move constructor and assignment on arrays"""
        for array_type in self.array_types:
            # We run this test only on non pointer arrays
            if not Test.is_array_ptr(array_type):
                test_func = getattr(test, "test_move_%s" % array_type)
                test_arrays = self._test_arrays(array_type)
                for test_array in test_arrays:
                    self.assertTrue(test_func(test_array))

    def test_value(self):
        """...Test value method of 1d arrays"""
        for array_type in self.array_types:
            # We run this test only on 1d arrays as value is not implemented
            # on others
            if Test.is_1d(array_type):
                test_func = getattr(test, "test_value_%s" % array_type)
                test_arrays = self._test_arrays(array_type)
                for test_array in test_arrays:
                    compare_array = Test.cast_1d_array_to_1d_dense(test_array)
                    # Then we compare that all extracted values are correct
                    for i in range(len(compare_array)):
                        self.assertEqual(
                            test_func(test_array, i), compare_array[i])

    def test_last(self):
        """...Test last method of 1d arrays"""
        for array_type in self.array_types:
            # We run this test only on 1d arrays as last is not implemented
            # on others
            if Test.is_1d(array_type):
                test_func = getattr(test, "test_last_%s" % array_type)
                test_arrays = self._test_arrays(array_type)
                for test_array in test_arrays:
                    compare_array = Test.cast_1d_array_to_1d_dense(test_array)
                    # Then we compare that the last value is correct
                    self.assertEqual(test_func(test_array), compare_array[-1])

    def test_dot(self):
        """...Test dot method of 1d arrays"""
        array_types_1d = [
            array_type for array_type in self.array_types
            if Test.is_1d(array_type)
        ]
        for array_type_1, array_type_2 in itertools.product(
                array_types_1d, array_types_1d):
            test_arrays_1 = self._test_arrays(array_type_1)
            test_arrays_2 = self._test_arrays(array_type_2)
            test_func = getattr(
                test, "test_dot_%s_%s" % (array_type_1, array_type_2))
            for test_array_1, test_array_2 in itertools.product(
                    test_arrays_1, test_arrays_2):
                compare_array_1 = Test.cast_1d_array_to_1d_dense(test_array_1)
                compare_array_2 = Test.cast_1d_array_to_1d_dense(test_array_2)
                self.assertAlmostEqual(
                    test_func(test_array_1, test_array_2),
                    compare_array_1.dot(compare_array_2))

    def test_as_array(self):
        """...Test behavior of as_array method"""
        test_arrays = self._test_arrays('BaseArrayDouble')
        original_arrays = self._test_arrays('BaseArrayDouble')
        for test_array, original_array in zip(test_arrays, original_arrays):
            self.assertAlmostEqual(test_array.sum(),
                                   test.test_as_array(test_array))
            # If it sparse data has been copied and original array not affected
            if hasattr(test_array, "toarray"):
                self.compare_arrays(test_array, original_array)
            # If it is dense data has been kept and modified
            else:
                self.compare_arrays(test_array, np.zeros_like(original_array))

    def test_as_array2d(self):
        """...Test behavior of as_array2d method"""
        test_arrays = self._test_arrays('BaseArrayDouble2d')
        original_arrays = self._test_arrays('BaseArrayDouble2d')
        for test_array, original_array in zip(test_arrays, original_arrays):
            self.assertEqual(test_array.sum(),
                             test.test_as_array2d(test_array))
            # If it sparse data has been copied and original array not affected
            if hasattr(test_array, "toarray"):
                self.compare_arrays(test_array, original_array)
            # If it is dense data has been kept and modified
            else:
                self.compare_arrays(test_array, np.zeros_like(original_array))

    def test_new_ptr(self):
        """...Test behavior of new_ptr method"""
        for array_type in self.array_types:
            # This test only concerns non ptr_type arrays
            # This method is not implemented on Base arrays yet
            if Test.is_array_ptr(array_type) or Test.is_base(array_type):
                continue
            test_func = getattr(test, "test_new_ptr_S%sPtr" % array_type)
            test_arrays = self._test_arrays(array_type)
            original_arrays = self._test_arrays(array_type)
            for test_array, original_array in zip(test_arrays,
                                                  original_arrays):
                self.assertAlmostEqual(original_array.sum(),
                                       test_func(test_array))
                # Data has been copied and original array should not be affected
                self.compare_arrays(test_array, original_array)

    def test_view(self):
        """...Test view method of arrays"""
        # We take two random numbers to have different test_arrays
        for array_type in self.array_types:
            # We run this test only on non pointer arrays
            if not Test.is_array_ptr(array_type):
                test_func = getattr(test, "test_view_%s" % array_type)
                test_arrays = self._test_arrays(array_type)
                for test_array in test_arrays:
                    test_array1 = test_array.copy()
                    test_array2 = test_array.copy()
                    test_array3 = test_array.copy()

                    expected = np.array([
                        test_array1.sum(),
                        test_array2.sum(),
                        test_array3.sum()
                    ])
                    results = test_func(test_array1, test_array2, test_array3)
                    np.testing.assert_array_almost_equal(results, expected)

                    Test.compare_arrays(test_array1, test_array)
                    Test.compare_arrays(test_array2,
                                        np.zeros_like(test_array2))
                    Test.compare_arrays(test_array3,
                                        np.zeros_like(test_array3))

    def test_slice_view1d(self):
        """...Test that view slicing works as expected on Array"""
        for test_array in self._test_arrays('ArrayDouble'):
            original_array = test_array.copy()
            start = 1
            end = 2

            full_view = original_array[:]
            start_slice_view = full_view[start:]
            end_slice_view = full_view[:-end]
            middle_slice_view = full_view[start:-end]
            start_start_slice_view = start_slice_view[start:]
            start_end_slice_view = end_slice_view[start:]
            end_start_slice_view = start_slice_view[:-end]
            end_end_slice_view = end_slice_view[:-end]

            # Check that every time the right view has been extracted
            expected = [
                full_view, start_slice_view, end_slice_view, middle_slice_view,
                start_start_slice_view, start_end_slice_view,
                end_start_slice_view, end_end_slice_view
            ]

            results = test.test_slice_view1d(test_array, start, end)
            for e, r in zip(expected, results):
                np.testing.assert_equal(r, e)

            # Check that the right part of the array has been set to 0
            output_array = original_array
            output_array[start:-end] = 0
            np.testing.assert_equal(test_array, output_array)

    def test_row_view(self):
        """...Test that row view works as expected on Array 2D"""
        for array_type in self.array_types:
            if Test.is_1d(array_type) or Test.is_array_ptr(array_type):
                continue

            test_func = getattr(test, "test_row_view_%s" % array_type)

            for test_array in self._test_arrays(array_type):
                original_array = test_array.copy()
                row = 1

                row_0 = original_array[0, :]
                row_1 = original_array[1, :]
                row_view = original_array[row, :]

                expected = [row_0, row_1, row_view]
                results = test_func(test_array, row)
                # Check that every time the right view has been extracted
                for e, r in zip(expected, results):
                    self.compare_arrays(r, self.cast_1d_array_to_1d_dense(e))

                # Check that the right part of the array has been set to 0
                output_array = self.cast_array_to_dense(original_array)
                output_array[row, :] = 0

                self.compare_arrays(test_array, output_array)

    def test_as_array_ptr(self):
        """...Test as array ptr method of arrays"""
        for array_type in self.array_types:
            # We run this test only on 1d arrays as last is not implemented
            # on others
            if Test.is_array_ptr(array_type):
                continue

            # To be done later ....
            # TODO: fix for others and then remove this line
            # What is missing for that ? Well, basically :
            # 1- Generate the test functions for the class BaseArrayDouble and BaseArrayDouble2d
            #   (See file array_test/src/array_test.h Macro TEST_AS_ARRAY_PTR
            #   and file array_test/src/array_test.cpp Macro TEST_AS_ARRAY_PTR_CPP)
            #
            # 2- The typemap_out of SArrayDouble2d, SSparseArray, SSparseArray2d,
            #   SBaseArrayDouble and SBaseArrayDouble2d
            #   (see file array/swig/sarray_typemap_out to see how the typemap out of
            #   SArrayDouble is written)
            if array_type != 'ArrayDouble':
                continue

            test_func = getattr(test, "test_as_array_ptr_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                np.testing.assert_equal(test_func(test_array), test_array)

    def test_sum(self):
        """...Test sum method of arrays"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_sum_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # Then we compare that the sum value is correct
                self.assertAlmostEqual(test_func(test_array), test_array.sum())

    def test_min(self):
        """...Test min method of arrays"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_min_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # Then we compare that the min value is correct
                self.assertEqual(test_func(test_array), test_array.min())

    def test_max(self):
        """...Test max method of arrays"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_max_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # Then we compare that the min value is correct
                self.assertEqual(test_func(test_array), test_array.max())

    def test_norm_sq(self):
        """...Test sum method of arrays"""
        for array_type in self.array_types:
            test_func = getattr(test, "test_norm_sq_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # We need this as we cannot take norm of sparse matrix
                norm_array = Test.cast_array_to_dense(test_array)

                # Then we compare that the norm value is correct
                self.assertAlmostEqual(
                    test_func(test_array),
                    norm(norm_array) ** 2)

    def test_multiply(self):
        """...Test *= operator of arrays"""
        scalar = np.sqrt(5)
        for array_type in self.array_types:
            test_func = getattr(test, "test_multiply_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # We need this as array will be multiplied in place
                original_array = test_array.copy()
                original_array *= scalar

                test_func(test_array, scalar)

                # Then we compare that the arrays are equal
                self.compare_arrays(test_array, original_array)

    def test_divide(self):
        """...Test /= operator of arrays"""
        scalar = np.sqrt(5)
        for array_type in self.array_types:
            test_func = getattr(test, "test_divide_%s" % array_type)
            test_arrays = self._test_arrays(array_type)
            for test_array in test_arrays:
                # We need this as array will be multiplied in place
                original_array = test_array.copy()
                original_array /= scalar

                test_func(test_array, scalar)

                # Then we compare that the arrays are equal
                self.compare_arrays(test_array, original_array)

    def test_arange(self):
        """...Test arange constructor of array"""
        min_max_couples = [(2, 5), (-10, 3), (-4, -2), (4, 2), (0, 0)]
        for min_value, max_value in min_max_couples:
            np.testing.assert_equal(
                test.test_arange(min_value, max_value),
                np.arange(min_value, max_value))

    def test_out_of_bound_errors(self):
        """...Test that we obtain an error for out of bounds indexes, if this
        test fails, check that debug=True in setup
        """
        # This test is commented as its stack trace is automatically printed
        # with self.assertRaisesRegex(ValueError, "BadIndex"):
        #     test.test_IndexError_ArrayDouble(np.zeros(10))
        #
        # with self.assertRaisesRegex(ValueError, "BadIndex"):
        #     test.test_IndexError_rows_ArrayDouble2d(np.zeros((10, 10)))
        #
        # with self.assertRaisesRegex(ValueError, "BadIndex"):
        #     test.test_IndexError_cols_ArrayDouble2d(np.zeros((10, 10)))

    def test_array_constructor(self):
        """...Test array constructors
        """
        self.assertEqual(test.test_constructor_ArrayDouble(4), 210)

    def test_sparsearray_constructor(self):
        """...Test Sparse array constructors
        """
        self.assertEqual(test.test_constructor_SparseArrayDouble(), 20)
        self.assertEqual(test.test_constructor_SparseArrayDouble2d(), 16)

    def test_BaseArray_empty_constructor(self):
        """...Test that we can create both sparse and non sparse arrays"""
        self.assertEqual(True, test.test_BaseArray_empty_constructor(True))
        self.assertEqual(False, test.test_BaseArray_empty_constructor(False))

    def test_varray_append(self):
        """...Test varray append functions
        """
        size = 12
        np.testing.assert_array_almost_equal(
            test.test_VArrayDouble_append1(size), np.arange(size))

        np.testing.assert_array_almost_equal(
            test.test_VArrayDouble_append(self.python_array_1,
                                          self.python_array_2),
            np.hstack((self.python_array_1, self.python_array_2)))

    def test_sort(self):
        """...Test sort method of arrays
        """
        a = np.random.normal(size=10)
        sorted_a = np.sort(a)
        decreasing_sorted_a = np.sort(a)[::-1]

        np.testing.assert_equal(test.test_sort_ArrayDouble(a, True), sorted_a)
        np.testing.assert_equal(
            test.test_sort_ArrayDouble(a, False), decreasing_sorted_a)

        test_a = a.copy()
        test.test_sort_inplace_ArrayDouble(test_a, True)
        np.testing.assert_equal(test_a, sorted_a)

        test_a = a.copy()
        test.test_sort_inplace_ArrayDouble(test_a, False)
        np.testing.assert_equal(test_a, decreasing_sorted_a)

    def test_sort_track_index(self):
        """...Test sort method of arrays which keep track of index
        """
        a = np.random.normal(size=10)
        sorted_index = np.argsort(a)
        sorted_a = np.sort(a)
        decreasing_sorted_index = np.argsort(a)[::-1]
        decreasing_sorted_a = np.sort(a)[::-1]

        index = np.empty_like(a, dtype=np.uint64)

        np.testing.assert_equal(
            test.test_sort_index_ArrayDouble(a, index, True), sorted_a)
        np.testing.assert_equal(index, sorted_index)

        np.testing.assert_equal(
            test.test_sort_index_ArrayDouble(a, index, False),
            decreasing_sorted_a)
        np.testing.assert_equal(index, decreasing_sorted_index)

        test_a = a.copy()
        test.test_sort_index_inplace_ArrayDouble(test_a, index, True)
        np.testing.assert_equal(test_a, sorted_a)
        np.testing.assert_equal(index, sorted_index)

        test_a = a.copy()
        test.test_sort_index_inplace_ArrayDouble(test_a, index, False)
        np.testing.assert_equal(test_a, decreasing_sorted_a)
        np.testing.assert_equal(index, decreasing_sorted_index)

    def test_sort_abs_track_index(self):
        """...Test sort in absolute value method of arrays which keep track of
        index
        """
        a = np.array([
            -0.45, -0.58, -1.31, -0.89, 0.31, -1.29, 1.77, -0.39, -1.2, -0.86
        ])
        sorted_index = np.array([4, 7, 0, 1, 9, 3, 8, 5, 2, 6])
        sorted_a = np.array([
            0.31, -0.39, -0.45, -0.58, -0.86, -0.89, -1.2, -1.29, -1.31, 1.77
        ])
        decreasing_sorted_index = sorted_index[::-1]
        decreasing_sorted_a = sorted_a[::-1]

        index = np.empty_like(a, dtype=np.uint64)
        test_a = a.copy()
        test.test_sort_abs_index_inplace_ArrayDouble(test_a, index, True)
        np.testing.assert_equal(test_a, sorted_a)
        np.testing.assert_equal(index, sorted_index)

        test_a = a.copy()
        test.test_sort_abs_index_inplace_ArrayDouble(test_a, index, False)
        np.testing.assert_equal(test_a, decreasing_sorted_a)
        np.testing.assert_equal(index, decreasing_sorted_index)

    def test_mult_incr(self):
        test_arrays = [
            self.python_array_1, self.python_array_2,
            self.python_sparse_array_1, self.python_sparse_array_2,
            self.python_sparse_array_3
        ]

        b = 3
        for test_array in test_arrays:
            array = self.python_array_1.copy()
            test.test_mult_incr_ArrayDouble(array, test_array, b)
            python_result = Test.cast_1d_array_to_1d_dense(
                self.python_array_1 + b * test_array)
            np.testing.assert_almost_equal(array, python_result)

    def test_mult_fill(self):
        test_arrays = [
            self.python_array_1, self.python_array_2,
            self.python_sparse_array_1, self.python_sparse_array_2,
            self.python_sparse_array_3
        ]
        b = 3
        for test_array in test_arrays:
            array = np.empty_like(self.python_array_1)
            test.test_mult_fill_ArrayDouble(array, test_array, b)
            python_result = Test.cast_1d_array_to_1d_dense(b * test_array)
            np.testing.assert_almost_equal(array, python_result)

    def test_mult_add_mult_incr(self):
        test_arrays = [
            self.python_array_1, self.python_array_2,
            self.python_sparse_array_1, self.python_sparse_array_2,
            self.python_sparse_array_3
        ]
        a = 3.14
        b = -2.76
        for x, y in itertools.product(test_arrays, test_arrays):
            array = self.python_array_1.copy()
            test.test_mult_add_mult_incr_ArrayDouble(array, x, a, y, b)
            python_result = Test.cast_1d_array_to_1d_dense(
                self.python_array_1 + a * x + b * y)
            np.testing.assert_almost_equal(array, python_result)


if __name__ == "__main__":
    unittest.main()
