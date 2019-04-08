# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxZero
from tick.prox.tests.prox import TestProx


class ProxZeroTest(object):
    def test_ProxZero(self):
        """...Test of ProxZero
        """
        coeffs = self.coeffs.copy().astype(self.dtype)
        out = coeffs.copy()

        prox = ProxZero().astype(self.dtype)
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-14)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)

        prox = ProxZero((3, 8)).astype(self.dtype)
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-14)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)


class ProxZeroTestFloat32(TestProx, ProxZeroTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxZeroTestFloat64(TestProx, ProxZeroTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
