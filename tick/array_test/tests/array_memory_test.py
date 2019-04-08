# License: BSD 3 clause

import gc
import unittest
import weakref

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from tick.array.build.array import tick_double_sparse2d_from_file
from tick.array.build.array import tick_double_sparse2d_to_file
from tick.array_test.build import array_test as test


class Test(unittest.TestCase):
    def test_varray_smart_pointer_in_cpp(self):
        """...Test C++ reference counter
        """
        vcc = test.VarrayContainer()
        self.assertEqual(vcc.nRef(), 0)
        vcc.initVarray()
        self.assertEqual(vcc.nRef(), 1)
        cu1 = test.VarrayUser()
        cu1.setArray(vcc)
        self.assertEqual(vcc.nRef(), 2)
        cu1.setArray(vcc)
        self.assertEqual(vcc.nRef(), 2)
        cu2 = test.VarrayUser()
        cu2.setArray(vcc)
        self.assertEqual(vcc.nRef(), 3)
        del cu1
        self.assertEqual(vcc.nRef(), 2)
        cu3 = test.VarrayUser()
        cu3.setArray(vcc)
        self.assertEqual(vcc.nRef(), 3)
        del cu3, cu2
        self.assertEqual(vcc.nRef(), 1)
        # we cannot check it will go to 0 after vcc deletion in Python

        cu4 = test.VarrayUser()
        cu4.setArray(vcc)
        self.assertEqual(vcc.nRef(), 2)
        del vcc
        self.assertEqual(cu4.nRef(), 1)
        # we cannot check it will go to 0 after cu4 deletion in Python
        del cu4

    def test_varray_smart_pointer_deletion1(self):
        """...Test that varray is still alive after deletion in Python
        """
        vcc = test.VarrayContainer()
        vcc.initVarray()
        # Now mix with some Python
        a = vcc.varrayPtr
        # This does not increment C++ reference counter
        self.assertEqual(vcc.nRef(), 1)
        # Get a weak ref of the array
        r = weakref.ref(a)
        del a
        np.testing.assert_array_almost_equal(r(), vcc.varrayPtr)
        del vcc
        self.assertIsNone(r())

    def test_varray_smart_pointer_deletion2(self):
        """...Test that base is deleted after a double assignment in Python
        """
        vcc = test.VarrayContainer()
        vcc.initVarray()

        a = vcc.varrayPtr
        b = vcc.varrayPtr
        r = weakref.ref(b)
        del a, vcc, b
        self.assertIsNone(r())

    def test_varray_smart_pointer_deletion3(self):
        """...Test that base is deleted after a double assignment in Python
        """
        vcc = test.VarrayContainer()
        vcc.initVarray()
        # Now mix with some Python
        a = vcc.varrayPtr
        a_sum = np.sum(a)
        # This does not increment C++ reference counter
        self.assertEqual(vcc.nRef(), 1)

        # Get a weak ref of the array
        r = weakref.ref(vcc.varrayPtr)

        del vcc
        np.testing.assert_array_almost_equal(a_sum, np.sum(a))

        self.assertIsNone(r())
        del a

    def test_sarray_memory_leaks(self):
        """...Test brute force method in order to see if we have a memory leak
        during typemap out
        """
        import os
        try:
            import psutil
        except ImportError:
            print('Without psutils we cannot ensure we have no memory leaks')
            return

        def get_memory_used():
            """Returns memory used by current process
            """
            process = psutil.Process(os.getpid())
            return process.memory_info()[0]

        initial_memory = get_memory_used()

        size = int(1e6)
        # The size in memory of an array of ``size`` doubles
        bytes_size = size * 8
        a = test.test_typemap_out_SArrayDoublePtr(size)
        first_filled_memory = get_memory_used()

        # Check that new memory is of the correct order (10%)
        self.assertAlmostEqual(first_filled_memory - initial_memory,
                               bytes_size, delta=1.1 * bytes_size)

        for _ in range(10):
            del a
            a = test.test_typemap_out_SArrayDoublePtr(size)
            filled_memory = get_memory_used()
            # Check memory is not increasing
            self.assertAlmostEqual(first_filled_memory - initial_memory,
                                   filled_memory - initial_memory,
                                   delta=1.1 * bytes_size)
        #print("\nfirst_filled_memory %.2g, filled_memory %.2g, initial_memory %.2g, array_bytes_size %.2g" % (first_filled_memory, filled_memory, initial_memory, bytes_size))

    def test_sarray_memory_leaks2(self):
        """...Test brute force method in order to see if we have a memory leak
        during typemap in or out
        """
        import os
        try:
            import psutil
        except ImportError:
            print('Without psutils we cannot ensure we have no memory leaks')
            return

        def get_memory_used():
            """Returns memory used by current process
            """
            process = psutil.Process(os.getpid())
            return process.memory_info()[0]

        size = int(1e6)
        a, b = np.ones(size), np.arange(size, dtype=float)

        initial_memory = get_memory_used()
        # The size in memory of an array of ``size`` doubles
        bytes_size = 2 * size * 8
        c = test.test_VArrayDouble_append(a, b)
        first_filled_memory = get_memory_used()

        # Check that new memory is of the correct order (10%)
        self.assertAlmostEqual(first_filled_memory,
                               initial_memory + bytes_size,
                               delta=1.1 * bytes_size)

        for _ in range(10):
            del c
            c = test.test_VArrayDouble_append(a, b)
            filled_memory = get_memory_used()
            # Check memory is not increasing
            self.assertAlmostEqual(first_filled_memory - initial_memory,
                                   filled_memory - initial_memory,
                                   delta=1.1 * bytes_size)

    def test_sarray2d_memory_leaks(self):
        """...Test brute force method in order to see if we have a memory leak
        during typemap out
        """
        import os
        try:
            import psutil
        except ImportError:
            print('Without psutils we cannot ensure we have no memory leaks')
            return

        def get_memory_used():
            """Returns memory used by current process
            """
            process = psutil.Process(os.getpid())
            return process.memory_info()[0]

        initial_memory = get_memory_used()

        n_rows = int(1e2)
        n_cols = int(1e3)
        # The size in memory of an array of ``size`` doubles
        bytes_size = n_rows * n_cols * 8
        a = test.test_typemap_out_SArrayDouble2dPtr(n_rows, n_cols)
        first_filled_memory = get_memory_used()

        # Check that new memory is of the correct order (10%)
        self.assertAlmostEqual(first_filled_memory - initial_memory,
                               bytes_size, delta=1.1 * bytes_size)

        for _ in range(10):
            del a
            a = test.test_typemap_out_SArrayDouble2dPtr(n_rows, n_cols)
            filled_memory = get_memory_used()
            # Check memory is not increasing
            self.assertAlmostEqual(first_filled_memory - initial_memory,
                                   filled_memory - initial_memory,
                                   delta=1.1 * bytes_size)

    def test_s_sparse_array2d_memory_leaks(self):
        """...Test brute force method in order to see if we have a memory leak
        during typemap out
        """
        import os

        try:
            import psutil
        except ImportError:
            print('Without psutils we cannot ensure we have no memory leaks')
            return

        def get_memory_used():
            """Returns memory used by current process
            """
            process = psutil.Process(os.getpid())
            return process.memory_info()[0]

        cereal_file = "sparse.gen.cereal"
        try:
            n_rows = int(1e3)
            n_cols = int(1e2)
            s_spar = int((n_rows * n_cols) * .3)

            data_size = (s_spar * 8)
            # The size in memory of an array of ``size`` doubles
            bytes_size = (data_size * 2) + ((n_rows + 1) * 8)

            sparsearray_double = scipy.sparse.rand(
                n_rows, n_cols, 0.3, format="csr", dtype=np.float64)

            tick_double_sparse2d_to_file(cereal_file, sparsearray_double)

            initial_memory = get_memory_used()
            a = tick_double_sparse2d_from_file(cereal_file)
            first_filled_memory = get_memory_used()

            # Check that new memory is of the correct order (10%)
            self.assertAlmostEqual(first_filled_memory - initial_memory,
                                   bytes_size, delta=1.1 * bytes_size)

            del a
            for i in range(10):
                # Check memory is not increasing
                gc.collect()
                filled_memory = get_memory_used()
                self.assertAlmostEqual(filled_memory, initial_memory,
                                       delta=1.1 * bytes_size)

                X = tick_double_sparse2d_from_file(cereal_file)
                del X

            gc.collect()

            end = get_memory_used()
            self.assertAlmostEqual(end, initial_memory, delta=1.1 * bytes_size)

        finally:
            if os.path.exists(cereal_file):
                os.remove(cereal_file)

    def test_varray_share_same_support(self):
        """...Test that modifications on Varray of in Python affect the same
        support
        """
        vcc = test.VarrayContainer()
        vcc.initVarray()
        # Now mix with some Python
        a = vcc.varrayPtr
        a[0] = 99.0
        self.assertEqual(vcc.varrayPtr[0], 99.0)
        vcc.varrayPtr[1] = 999.0
        self.assertEqual(a[1], 999.0)

    def test_sbasearrayptr(self):
        sparsearray_double = csr_matrix(
            (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]),
             np.array([0, 5])), shape=(1, 12))
        test.test_sbasearray_container_new(sparsearray_double)
        self.assertEqual(test.test_sbasearray_container_compute(), 45)
        test.test_sbasearray_container_clear()
        self.assertEqual(test.test_sbasearray_container_compute(), -1)
        array_double = np.arange(2, 14, dtype=float)
        test.test_sbasearray_container_new(array_double)
        self.assertEqual(test.test_sbasearray_container_compute(),
                         array_double.sum())
        test.test_sbasearray_container_clear()
        self.assertEqual(test.test_sbasearray_container_compute(), -1)

    def test_ref_sbasearrayptr(self):
        sparsearray_double = csr_matrix(
            (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 8, 10]),
             np.array([0, 5])), shape=(1, 12))
        refdata = weakref.ref(sparsearray_double.data)
        refindices = weakref.ref(sparsearray_double.indices)
        refindptr = weakref.ref(sparsearray_double.indptr)
        test.test_sbasearray_container_new(sparsearray_double)
        del sparsearray_double
        self.assertIsNone(refindptr())
        self.assertIsNotNone(refdata())
        self.assertIsNotNone(refindices())
        test.test_sbasearray_container_clear()
        self.assertIsNone(refdata())
        self.assertIsNone(refindices())

        array_double = np.arange(2, 14, dtype=float)
        ref = weakref.ref(array_double)
        test.test_sbasearray_container_new(array_double)
        del array_double
        self.assertIsNotNone(ref())
        test.test_sbasearray_container_clear()
        self.assertIsNone(ref())

    def test_sbasearray2dptr(self):
        sparsearray2d_double = csr_matrix(
            (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 1, 3]),
             np.array([0, 3, 5])), shape=(2, 4))
        test.test_sbasearray2d_container_new(sparsearray2d_double)
        self.assertEqual(test.test_sbasearray2d_container_compute(), 39)
        test.test_sbasearray2d_container_clear()
        self.assertEqual(test.test_sbasearray2d_container_compute(), -1)
        array2d_double = np.array([[1.2, 3], [4, 5]])
        test.test_sbasearray2d_container_new(array2d_double)
        self.assertEqual(test.test_sbasearray2d_container_compute(),
                         array2d_double.sum())
        test.test_sbasearray2d_container_clear()
        self.assertEqual(test.test_sbasearray2d_container_compute(), -1)

    def test_ref_sbasearray2dptr(self):
        sparsearray2d_double = csr_matrix(
            (np.array([1., 2, 3, 4, 5]), np.array([2, 4, 6, 1, 3]),
             np.array([0, 3, 5])), shape=(2, 4))
        refdata = weakref.ref(sparsearray2d_double.data)
        refindices = weakref.ref(sparsearray2d_double.indices)
        refindptr = weakref.ref(sparsearray2d_double.indptr)
        test.test_sbasearray2d_container_new(sparsearray2d_double)
        del sparsearray2d_double
        self.assertIsNotNone(refindptr())
        self.assertIsNotNone(refdata())
        self.assertIsNotNone(refindices())
        test.test_sbasearray2d_container_clear()
        self.assertIsNone(refindptr())
        self.assertIsNone(refdata())
        self.assertIsNone(refindices())

        array2d_double = np.array([[1.2, 3], [4, 5]])
        ref = weakref.ref(array2d_double)
        test.test_sbasearray2d_container_new(array2d_double)
        del array2d_double
        self.assertIsNotNone(ref())
        test.test_sbasearray2d_container_clear()
        self.assertIsNone(ref())


if __name__ == "__main__":
    unittest.main()
