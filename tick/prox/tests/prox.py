# License: BSD 3 clause

import numpy as np
import unittest


class TestProx(unittest.TestCase):
    def setUp(self):
        self.coeffs = np.array([
            -0.86017247, -0.58127151, -0.6116414, 0.23186939, -0.85916332,
            1.6783094, 1.39635801, 1.74346116, -0.27576309, -1.00620197
        ])

    def __init__(self, *args, dtype="float64", **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dtype = dtype
        self.decimal_places = 7
        self.delta = 1e-15
        if np.dtype(self.dtype) == np.dtype("float32"):
            self.decimal_places = 3
            self.delta = 1e-6
