# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxPositive
from tick.prox.tests.prox import TestProx


class ProxPositiveTest(object):
    def test_ProxPositive(self):
        """...Test of ProxPositive
        """
        coeffs = self.coeffs.copy().astype(self.dtype)

        prox = ProxPositive().astype(self.dtype)
        out = coeffs.copy()
        out[out < 0] = 0
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=self.delta)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)

        prox = ProxPositive((3, 8)).astype(self.dtype)
        out = coeffs.copy()
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=self.delta)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)


class ProxPositiveTestFloat32(TestProx, ProxPositiveTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxPositiveTestFloat64(TestProx, ProxPositiveTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
