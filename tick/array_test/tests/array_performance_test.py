# License: BSD 3 clause

# -*- coding: utf8 -*-

import unittest
import numpy as np

from tick.array_test.build.array_test import test_sum_double_pointer, \
    test_sum_ArrayDouble, test_sum_SArray_shared_ptr, test_sum_VArray_shared_ptr
"""
ref_size = 10000
ref_n_loops = 10000
start = time.process_time()
ref_result = test_sum_double_pointer(ref_size, ref_n_loops)
end = time.process_time()
ref_needed_time = end - start
"""


class Test(unittest.TestCase):
    """
    def test_array_speed(self):
        \"""...Test speed of ArrayDouble is equivalent to a double pointer array
        \"""
        start = time.process_time()
        result = test_sum_ArrayDouble(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.2)

    def test_sarrayptr_speed(self):
        \"""...Test speed of SArrayDoublePtr is equivalent to a double pointer
        array
        \"""
        start = time.process_time()
        result = test_sum_SArray_shared_ptr(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.2)

    def test_varrayptr_speed(self):
        \"""...Test speed of VArrayDoublePtr is equivalent to a double pointer
        array
        \"""
        start = time.process_time()
        result = test_sum_VArray_shared_ptr(ref_size, ref_n_loops)
        end = time.process_time()
        needed_time = end - start
        self.assertEqual(result, ref_result)
        if needed_time > ref_needed_time:
            np.testing.assert_allclose(needed_time, ref_needed_time, rtol=0.1)
    """


if __name__ == "__main__":
    unittest.main()
