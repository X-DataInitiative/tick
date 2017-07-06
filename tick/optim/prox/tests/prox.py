# License: BSD 3 clause

import numpy as np
import unittest


class TestProx(unittest.TestCase):
    def setUp(self):
        self.coeffs = np.array([-0.86017247, -0.58127151, -0.6116414,
                                0.23186939, -0.85916332, 1.6783094,
                                1.39635801, 1.74346116, -0.27576309,
                                -1.00620197])
