# License: BSD 3 clause

# -*- coding: utf8 -*-

import unittest
import numpy as np
from scipy.sparse import csr_matrix
from tick.array_test.build import array_test as test


class Test(unittest.TestCase):
    def setUp(self):

        self.correspondence_dict = {
            'Double': {
                'python_type': float,
                'cpp_type': 'double',
            },
            'Float': {
                'python_type': np.float32,
                'cpp_type': 'float',
            },
            'Int': {
                'python_type': np.int32,
                'cpp_type': 'std::int32_t',
            },
            'UInt': {
                'python_type': np.uint32,
                'cpp_type': 'std::uint32_t',
            },
            'Short': {
                'python_type': np.int16,
                'cpp_type': 'std::int16_t',
            },
            'UShort': {
                'python_type': np.uint16,
                'cpp_type': 'std::uint16_t',
            },
            'Long': {
                'python_type': np.int64,
                'cpp_type': 'std::int64_t',
            },
            'ULong': {
                'python_type': np.uint64,
                'cpp_type': 'std::uint64_t',
            }
        }

        for array_type, info in self.correspondence_dict.items():
            # fixed number of the correct type
            info['number'] = 148  # info['python_type'](np.exp(5))

            # The dense array of the corresponding type
            python_array = np.array([1, 2, 5, 0, 4,
                                     1]).astype(info['python_type'])

            # The dense array 2D of the corresponding type
            python_array_2d = np.array([[1, 2, 5],
                                        [0, 4, 1]]).astype(info['python_type'])

            # The list of dense array of the corresponding type
            python_array_list_1d = [
                np.array([1, 2, 5]).astype(info['python_type']),
                np.array([1, 2, 9]).astype(info['python_type']),
                np.array([0, 4]).astype(info['python_type'])
            ]

            python_array2d_list_1d = [
                np.array([[1, 2, 5], [0, 4, 1]]).astype(info['python_type']),
                np.array([[1, 2, 9], [1, 2, 5],
                          [0, 4, 1]]).astype(info['python_type']),
                np.array([[0]]).astype(info['python_type'])
            ]

            python_array_list_2d = [[
                np.array([1, 2, 5]).astype(info['python_type'])
            ], [
                np.array([1, 2, 9, 5]).astype(info['python_type']),
                np.array([0, 4, 1]).astype(info['python_type'])
            ], []]

            python_array2d_list_2d = [[
                np.array([[1, 2, 5], [0, 4, 1]]).astype(info['python_type']),
            ], [
                np.array([[1, 2, 9], [1, 2, 5],
                          [0, 4, 1]]).astype(info['python_type']),
                np.array([[0]]).astype(info['python_type'])
            ], []]

            # The sparse array of the corresponding type
            python_sparse_array = csr_matrix(
                (np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 9]),
                 np.array([0, 4]))).astype(info['python_type'])

            # The sparse array 2D of the corresponding type
            python_sparse_array_2d = csr_matrix(
                (np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 4]),
                 np.array([0, 3, 4]))).astype(info['python_type'])

            python_sparse_array_list_1d = [
                csr_matrix((np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 9]),
                            np.array([0, 4]))).astype(info['python_type']),
                csr_matrix((np.array([1.5, 2, 3]), np.array([3, 5, 11]),
                            np.array([0, 3]))).astype(info['python_type']),
                csr_matrix((np.array([1.5, 2, 3, 1, 2]),
                            np.array([3, 5, 7, 4, 1]),
                            np.array([0, 5]))).astype(info['python_type'])
            ]

            # TODO: add mixed list

            python_sparse_array2d_list_1d = [
                csr_matrix((np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 9]),
                            np.array([0, 3, 4]))).astype(info['python_type']),
                csr_matrix((np.array([1.5, 2, 3]), np.array([3, 5, 11]),
                            np.array([0, 1, 3]))).astype(info['python_type']),
                csr_matrix(
                    (np.array([1.5, 2, 3, 1, 2]), np.array([3, 5, 7, 4, 1]),
                     np.array([0, 2, 3, 5]))).astype(info['python_type'])
            ]

            python_sparse_array_list_2d = [[
                csr_matrix((np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 9]),
                            np.array([0, 4]))).astype(info['python_type'])
            ], [
                csr_matrix((np.array([1.5, 2, 3]), np.array([3, 5, 11]),
                            np.array([0, 3]))).astype(info['python_type']),
                csr_matrix((np.array([1.5, 2, 3, 1, 2]),
                            np.array([3, 5, 7, 4, 1]),
                            np.array([0, 5]))).astype(info['python_type'])
            ], []]

            python_sparse_array2d_list_2d = [[
                csr_matrix((np.array([1.5, 2, 3, 1]), np.array([3, 5, 7, 9]),
                            np.array([0, 3, 4]))).astype(info['python_type'])
            ], [
                csr_matrix((np.array([1.5, 2, 3]), np.array([3, 5, 11]),
                            np.array([0, 1, 3]))).astype(info['python_type']),
                csr_matrix(
                    (np.array([1.5, 2, 3, 1, 2]), np.array([3, 5, 7, 4, 1]),
                     np.array([0, 2, 3, 5]))).astype(info['python_type'])
            ], []]

            info['python_array'] = python_array
            info['python_array_2d'] = python_array_2d
            info['python_array_list_1d'] = python_array_list_1d
            info['python_array2d_list_1d'] = python_array2d_list_1d
            info['python_array2d_list_2d'] = python_array2d_list_2d
            info['python_array_list_2d'] = python_array_list_2d
            info['python_sparse_array'] = python_sparse_array
            info['python_sparse_array_2d'] = python_sparse_array_2d
            info['python_sparse_array_list_1d'] = python_sparse_array_list_1d
            info['python_sparse_array2d_list_1d'] = \
                python_sparse_array2d_list_1d
            info['python_sparse_array_list_2d'] = python_sparse_array_list_2d
            info['python_sparse_array2d_list_2d'] = \
                python_sparse_array2d_list_2d

            # corresponding test functions
            # for typemap in
            info['typemap_in_array'] = \
                getattr(test, 'test_typemap_in_Array%s' % array_type)
            info['typemap_in_array_2d'] = \
                getattr(test, 'test_typemap_in_Array%s2d' % array_type)
            info['typemap_in_array_list_1d'] = \
                getattr(test, 'test_typemap_in_Array%sList1D' % array_type)
            info['typemap_in_array_list_2d'] = \
                getattr(test, 'test_typemap_in_Array%sList2D' % array_type)

            info['typemap_in_sparse_array'] = \
                getattr(test, 'test_typemap_in_SparseArray%s' % array_type)
            info['typemap_in_sparse_array_2d'] = \
                getattr(test, 'test_typemap_in_SparseArray%s2d' % array_type)

            info['typemap_in_sarray_ptr'] = \
                getattr(test, 'test_typemap_in_SArray%sPtr' % array_type)
            info['typemap_in_sarray_ptr_2d'] = \
                getattr(test, 'test_typemap_in_SArray%s2dPtr' % array_type)
            info['typemap_in_sarray_ptr_list_1d'] = \
                getattr(test, 'test_typemap_in_SArray%sPtrList1D' % array_type)
            info['typemap_in_sarray_ptr_list_2d'] = \
                getattr(test, 'test_typemap_in_SArray%sPtrList2D' % array_type)
            info['typemap_in_sarray2d_ptr_list_1d'] = \
                getattr(test, 'test_typemap_in_SArray%s2dPtrList1D' %
                        array_type)
            info['typemap_in_sarray2d_ptr_list_2d'] = \
                getattr(test, 'test_typemap_in_SArray%s2dPtrList2D' %
                        array_type)

            info['typemap_in_varray_ptr'] = \
                getattr(test, 'test_typemap_in_VArray%sPtr' % array_type)
            info['typemap_in_varray_ptr_list_1d'] = \
                getattr(test, 'test_typemap_in_VArray%sPtrList1D' % array_type)
            info['typemap_in_varray_ptr_list_2d'] = \
                getattr(test, 'test_typemap_in_VArray%sPtrList2D' % array_type)

            info['typemap_in_base_array'] = \
                getattr(test, 'test_typemap_in_BaseArray%s' % array_type)
            info['typemap_in_base_array_2d'] = \
                getattr(test, 'test_typemap_in_BaseArray%s2d' % array_type)

            info['typemap_in_sparse_array_ptr'] = \
                getattr(test, 'test_typemap_in_SSparseArray%sPtr' % array_type)
            info['typemap_in_sparse_array_2d_ptr'] = \
                getattr(test, 'test_typemap_in_SSparseArray%s2dPtr' %
                        array_type)

            info['typemap_in_base_array_ptr'] = \
                getattr(test, 'test_typemap_in_SBaseArray%sPtr' % array_type)
            info['typemap_in_base_array_2d_ptr'] = \
                getattr(test, 'test_typemap_in_SBaseArray%s2dPtr' %
                        array_type)
            info['typemap_in_base_array_list_1d'] = \
                getattr(test, 'test_typemap_in_BaseArray%sList1D' % array_type)
            info['typemap_in_base_array_list_2d'] = \
                getattr(test, 'test_typemap_in_BaseArray%sList2D' % array_type)
            info['typemap_in_base_array2d_list_1d'] = \
                getattr(test, 'test_typemap_in_BaseArray%s2dList1D' %
                        array_type)
            info['typemap_in_base_array2d_list_2d'] = \
                getattr(test, 'test_typemap_in_BaseArray%s2dList2D' %
                        array_type)

            info['typemap_in_base_array_ptr_list_1d'] = \
                getattr(test, 'test_typemap_in_SBaseArray%sPtrList1D' %
                        array_type)
            info['typemap_in_base_array_ptr_list_2d'] = \
                getattr(test, 'test_typemap_in_SBaseArray%sPtrList2D' %
                        array_type)
            info['typemap_in_base_array2d_ptr_list_1d'] = \
                getattr(test, 'test_typemap_in_SBaseArray%s2dPtrList1D' %
                        array_type)
            info['typemap_in_base_array2d_ptr_list_2d'] = \
                getattr(test, 'test_typemap_in_SBaseArray%s2dPtrList2D' %
                        array_type)

            # Functions that are not overloaded to test error messages
            info['typemap_in_array_not_ol'] = \
                getattr(test, 'test_typemap_in_not_ol_Array%s' % array_type)
            info['typemap_in_array_2d_not_ol'] = \
                getattr(test, 'test_typemap_in_not_ol_Array%s2d' % array_type)
            info['typemap_in_sparse_array_not_ol'] = \
                getattr(test,
                        'test_typemap_in_not_ol_SparseArray%s' % array_type)
            info['typemap_in_base_array_not_ol'] = \
                getattr(test,
                        'test_typemap_in_not_ol_BaseArray%s' % array_type)

            info['typemap_in_array_list_1d_not_ol'] = \
                getattr(test,
                        'test_typemap_in_not_ol_Array%sList1D' % array_type)
            info['typemap_in_array_list_2d_not_ol'] = \
                getattr(test,
                        'test_typemap_in_not_ol_Array%sList2D' % array_type)

            # for typemap out
            info['typemap_out_sarray_ptr'] = \
                getattr(test, 'test_typemap_out_SArray%sPtr' % array_type)
            info['typemap_out_sarray_ptr_list_1d'] = \
                getattr(test, 'test_typemap_out_SArray%sPtrList1D' % array_type)
            info['typemap_out_sarray_ptr_list_2d'] = \
                getattr(test, 'test_typemap_out_SArray%sPtrList2D' % array_type)
            info['typemap_out_sarray_2d_ptr'] = \
                getattr(test, 'test_typemap_out_SArray%s2dPtr' % array_type)

    def test_array_typemap_in(self):
        """...Test we can pass an Array as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array']
            extract_function = info['typemap_in_array']
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_array2d_typemap_in(self):
        """...Test we can pass an Array2d as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_2d']
            extract_function = info['typemap_in_array_2d']
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_array_list_1d_typemap_in(self):
        """...Test we can pass a list of Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_1d']
            extract_function = info['typemap_in_array_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_array_list_2d_typemap_in(self):
        """...Test we can pass a list of list of Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_2d']
            extract_function = info['typemap_in_array_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sparsearray_typemap_in(self):
        """...Test we pass a SparseArray as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_sparse_array = info['python_sparse_array']
            extract_function = info['typemap_in_sparse_array']
            self.assertEqual(python_sparse_array.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sparsearray2d_typemap_in(self):
        """...Test we can pass a SparseArray2d as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_sparse_array_2d = info['python_sparse_array_2d']
            extract_function = info['typemap_in_sparse_array_2d']
            self.assertEqual(python_sparse_array_2d.sum(),
                             extract_function(python_sparse_array_2d))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray_ptr_typemap_in(self):
        """...Test we can pass an SArray shared pointer as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array']
            extract_function = info['typemap_in_sarray_ptr']
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray2dptr_typemap_in(self):
        """...Test we can pass an SArray2d shared pointer as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_2d']
            extract_function = info['typemap_in_sarray_ptr_2d']
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray_ptr_list_1d_typemap_in(self):
        """...Test we can pass a list of SArray shared pointers as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_1d']
            extract_function = info['typemap_in_sarray_ptr_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of list of SArray shared pointers as
        argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_2d']
            extract_function = info['typemap_in_sarray_ptr_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray2d_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of SArray2d shared pointers as
        argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_1d']
            extract_function = info['typemap_in_sarray2d_ptr_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarray2d_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of list of SArray2d shared pointers as
        argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_2d']
            extract_function = info['typemap_in_sarray2d_ptr_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_varray_ptr_typemap_in(self):
        """...Test we can pass an VArray shared pointer as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array']
            extract_function = info['typemap_in_varray_ptr']
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_varray_ptr_list_1d_typemap_in(self):
        """...Test we can pass a list of VArray shared pointers as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_1d']
            extract_function = info['typemap_in_varray_ptr_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_varray_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of list of VArray shared pointers as
        argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_2d']
            extract_function = info['typemap_in_varray_ptr_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray_typemap_in(self):
        """...Test we can pass an BaseArray as argument for sparse and dense
        arrays
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array']
            python_sparse_array = info['python_sparse_array']
            extract_function = info['typemap_in_base_array']
            # Test dense
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            # Test sparse
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray2d_typemap_in(self):
        """...Test we can pass an BaseArray2d as argument for sparse and dense
        arrays
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_2d']
            python_sparse_array = info['python_sparse_array_2d']
            extract_function = info['typemap_in_base_array_2d']
            # Test dense
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            # Test sparse
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_ssparsearrayptr_typemap_in(self):
        """...Test we can pass a SSparseArray shared pointer as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_sparse_array = info['python_sparse_array']
            extract_function = info['typemap_in_sparse_array_ptr']
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_ssparsearray2d_ptr_typemap_in(self):
        """...Test we can pass a SSparseArray2d shared pointer as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_sparse_array = info['python_sparse_array_2d']
            extract_function = info['typemap_in_sparse_array_2d_ptr']
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sbasearray_ptr_typemap_in(self):
        """...Test we can pass an BaseArray as argument for sparse and dense
        arrays
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array']
            python_sparse_array = info['python_sparse_array']
            extract_function = info['typemap_in_base_array_ptr']
            # Test dense
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            # Test sparse
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sbasearray2d_ptr_typemap_in(self):
        """...Test we can pass an BaseArray2d as argument for sparse and dense
        arrays
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_2d']
            python_sparse_array = info['python_sparse_array_2d']
            extract_function = info['typemap_in_base_array_2d_ptr']
            # Test dense
            self.assertEqual(python_array.sum(),
                             extract_function(python_array))
            # Test sparse
            self.assertEqual(python_sparse_array.data.sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray_list_1d_typemap_in(self):
        """...Test we can pass a list of base Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_1d']
            python_sparse_array = info['python_sparse_array_list_1d']
            extract_function = info['typemap_in_base_array_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray_list_2d_typemap_in(self):
        """...Test we can pass a list of list of base Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_2d']
            python_sparse_array = info['python_sparse_array_list_2d']
            extract_function = info['typemap_in_base_array_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0][0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray2d_list_1d_typemap_in(self):
        """...Test we can pass a list of base Arrays 2D as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_1d']
            python_sparse_array = info['python_sparse_array2d_list_1d']
            extract_function = info['typemap_in_base_array2d_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray2d_list_2d_typemap_in(self):
        """...Test we can pass a list of list of base Arrays 2D as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_2d']
            python_sparse_array = info['python_sparse_array2d_list_2d']
            extract_function = info['typemap_in_base_array2d_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0][0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray_ptr_list_1d_typemap_in(self):
        """...Test we can pass a list of base Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_1d']
            python_sparse_array = info['python_sparse_array_list_1d']
            extract_function = info['typemap_in_base_array_ptr_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of list of base Arrays as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array_list_2d']
            python_sparse_array = info['python_sparse_array_list_2d']
            extract_function = info['typemap_in_base_array_ptr_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0][0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray2d_ptr_list_1d_typemap_in(self):
        """...Test we can pass a list of base Arrays 2D as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_1d']
            python_sparse_array = info['python_sparse_array2d_list_1d']
            extract_function = info['typemap_in_base_array2d_ptr_list_1d']
            self.assertEqual(python_array[0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_basearray2d_ptr_list_2d_typemap_in(self):
        """...Test we can pass a list of list of base Arrays 2D as argument
        """
        for array_type, info in self.correspondence_dict.items():
            python_array = info['python_array2d_list_2d']
            python_sparse_array = info['python_sparse_array2d_list_2d']
            extract_function = info['typemap_in_base_array2d_ptr_list_2d']
            self.assertEqual(python_array[0][0].sum(),
                             extract_function(python_array))
            self.assertEqual(python_sparse_array[0][0].sum(),
                             extract_function(python_sparse_array))
            self.assertEqual(info['number'], extract_function(info['number']))

    def test_sarrayptr_typemap_out(self):
        """...Test we can return an SArray shared pointer
        """
        size = 10
        for array_type, info in self.correspondence_dict.items():
            python_array = np.arange(size, dtype=info['python_type'])
            extract_function = info['typemap_out_sarray_ptr']
            np.testing.assert_equal(python_array, extract_function(size))

    def test_sarrayptr_list1d_typemap_out(self):
        """...Test we can return a list of SArray shared pointer
        """
        size = 10
        for array_type, info in self.correspondence_dict.items():
            python_array = [
                i * np.ones(i, dtype=info['python_type']) for i in range(size)
            ]
            extract_function = info['typemap_out_sarray_ptr_list_1d']
            np.testing.assert_equal(python_array, extract_function(size))

    def test_sarrayptr_list2d_typemap_out(self):
        """...Test we can return an SArray shared pointer
        """
        size1 = 10
        size2 = 10

        for array_type, info in self.correspondence_dict.items():
            python_array = [[
                i * np.ones(i, dtype=info['python_type']) for i in range(j)
            ] for j in range(size2)]
            extract_function = info['typemap_out_sarray_ptr_list_2d']

            np.testing.assert_equal(python_array, extract_function(
                size1, size2))

    def test_sarray2dptr_typemap_out(self):
        """...Test we can return an SArray2d shared pointer
        """
        row_size = 6
        col_size = 4
        for array_type, info in self.correspondence_dict.items():
            python_array = np.arange(row_size * col_size,
                                     dtype=info['python_type']).reshape(
                                         row_size, col_size)
            extract_function = info['typemap_out_sarray_2d_ptr']
            np.testing.assert_equal(python_array,
                                    extract_function(row_size, col_size))

    def test_basearray_errors_typemap_in(self):
        """...Test errors raised by typemap in on BaseArray
        """
        for array_type, info in self.correspondence_dict.items():
            extract_function = info['typemap_in_base_array_not_ol']

            # Test if we pass something that has no link with a numpy array
            regex_error_random = "Expecting.*1d.*%s.*array.*sparse" % \
                                 info['cpp_type']
            self.assertRaisesRegex(ValueError, regex_error_random,
                                   extract_function, object())

            # Test if we pass an array of another type
            regex_error_wrong_type = "Expecting.*%s.*array" % info['cpp_type']
            other_array_type = [
                key for key in self.correspondence_dict.keys()
                if key != array_type
            ][0]
            python_other_type_array = self.correspondence_dict[
                other_array_type]['python_array']

            self.assertRaisesRegex(ValueError, regex_error_wrong_type,
                                   extract_function, python_other_type_array)

    def test_sparsearray_errors_typemap_in(self):
        """...Test errors raised by typemap in on SparseArray
        """
        for array_type, info in self.correspondence_dict.items():
            extract_function = info['typemap_in_sparse_array_not_ol']

            regex_error_dense = "Expecting.*sparse"
            python_array = info['python_array']
            self.assertRaisesRegex(ValueError, regex_error_dense,
                                   extract_function, python_array)

            # Test if we pass an array of another type
            regex_error_wrong_type = "Expecting.*%s.*array" % info['cpp_type']
            other_array_type = [
                key for key in self.correspondence_dict.keys()
                if key != array_type
            ][0]
            python_other_type_array = self.correspondence_dict[
                other_array_type]['python_sparse_array']

            self.assertRaisesRegex(ValueError, regex_error_wrong_type,
                                   extract_function, python_other_type_array)

    def test_array_errors_typemap_in(self):
        """...Test errors raised by typemap in on SparseArray
        """
        for array_type, info in self.correspondence_dict.items():
            extract_function = info['typemap_in_array_not_ol']

            # Test if we pass something that has no link with a numpy array
            regex_error_random = "Expecting(.*?)numpy(.*?)(a|A)rray"
            self.assertRaisesRegex(ValueError, regex_error_random,
                                   extract_function, object())

            regex_error_sparse = "Expecting.*dense"
            python_sparse_array = info['python_sparse_array']
            self.assertRaisesRegex(ValueError, regex_error_sparse,
                                   extract_function, python_sparse_array)

            # Test if we pass an array of another type
            regex_error_wrong_type = "Expecting.*%s.*array" % info['cpp_type']
            other_array_type = [
                key for key in self.correspondence_dict.keys()
                if key != array_type
            ][0]
            python_other_type_array = self.correspondence_dict[
                other_array_type]['python_array']

            self.assertRaisesRegex(ValueError, regex_error_wrong_type,
                                   extract_function, python_other_type_array)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                non_contiguous = np.arange(20).astype(float)[::2]
                extract_function(non_contiguous)

            with self.assertRaisesRegex(ValueError, "dimensional"):
                extract_function(np.zeros((5, 5)))

            with self.assertRaisesRegex(ValueError, "dimensional"):
                extract_function(np.zeros((5, 5, 5)))

    def test_array_2d_type_error(self):
        """...Test array2d interfacing between SWIG and Python
        """
        for array_type, info in self.correspondence_dict.items():
            extract_function = info['typemap_in_array_2d_not_ol']

            with self.assertRaisesRegex(ValueError, "contiguous"):
                non_contiguous = np.arange(2, 17, dtype="double").reshape(
                    3, 5)[::2]
                extract_function(non_contiguous)

            with self.assertRaisesRegex(ValueError, "dimensional"):
                extract_function(
                    np.arange(0, 20, dtype="double").reshape(2, 2, 5))

    def test_array_list_type_error(self):
        """...Test errors raised by typemap in on Array list 1d and 2d
        """
        for array_type, info in self.correspondence_dict.items():
            extract_function = info['typemap_in_array_list_1d_not_ol']
            with self.assertRaisesRegex(ValueError, "list(.*?)(a|A)rray"):
                extract_function(np.ones((5, 5)))

            extract_function = info['typemap_in_array_list_2d_not_ol']
            with self.assertRaisesRegex(ValueError,
                                        "2d(.*?)list(.*?)(a|A)rray"):
                extract_function(np.ones((5, 5, 5)))


if __name__ == "__main__":
    unittest.main()
