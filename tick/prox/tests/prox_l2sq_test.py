# License: BSD 3 clause

import unittest

import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_almost_equal

from tick.prox import ProxL2Sq
from tick.prox.tests.prox import TestProx


class ProxL2SqTest(object):
    def test_ProxL2Sq(self):
        """...Test of ProxL2Sq
        """
        coeffs = self.coeffs.copy().astype(self.dtype)

        l_l2sq = 3e-2
        t = 1.7

        prox = ProxL2Sq(l_l2sq).astype(self.dtype)
        out = coeffs.copy()
        out *= 1. / (1. + t * l_l2sq)
        self.assertAlmostEqual(
            prox.value(coeffs), 0.5 * l_l2sq * norm(coeffs) ** 2.,
            delta=self.delta)
        assert_almost_equal(
            prox.call(coeffs, step=t), out, decimal=self.decimal_places)

        prox = ProxL2Sq(l_l2sq, (3, 8)).astype(self.dtype)
        out = coeffs.copy()
        out[3:8] *= 1. / (1. + t * l_l2sq)
        self.assertAlmostEqual(
            prox.value(coeffs), 0.5 * l_l2sq * norm(coeffs[3:8]) ** 2.,
            delta=self.delta)
        assert_almost_equal(
            prox.call(coeffs, step=t), out, decimal=self.decimal_places)

        prox = ProxL2Sq(l_l2sq, (3, 8), positive=True).astype(self.dtype)
        out = coeffs.copy()
        out[3:8] *= 1. / (1. + t * l_l2sq)
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        self.assertAlmostEqual(
            prox.value(coeffs), 0.5 * l_l2sq * norm(coeffs[3:8]) ** 2.,
            delta=self.delta)
        assert_almost_equal(
            prox.call(coeffs, step=t), out, decimal=self.decimal_places)

        prox = ProxL2Sq(l_l2sq, (3, 8)).astype(self.dtype)
        out = coeffs.copy()
        t = np.linspace(1, 10, 5).astype(self.dtype)
        out[3:8] *= 1. / (1. + t * l_l2sq)
        self.assertAlmostEqual(
            prox.value(coeffs), 0.5 * l_l2sq * norm(coeffs[3:8]) ** 2.,
            delta=self.delta)
        assert_almost_equal(
            prox.call(coeffs, t), out, decimal=self.decimal_places)


class ProxL2SqTestFloat32(TestProx, ProxL2SqTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxL2SqTestFloat64(TestProx, ProxL2SqTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
