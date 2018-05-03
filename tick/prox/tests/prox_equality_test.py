# License: BSD 3 clause

import unittest

import numpy as np

from tick.prox import ProxEquality
from tick.prox.tests.prox import TestProx


class ProxEqualityTest(object):
    def test_ProxEquality(self):
        """...Test of ProxEquality
        """
        coeffs = self.coeffs.copy()
        strength = 0.5

        prox = ProxEquality(strength)
        self.assertIsNone(prox.strength)
        prox.strength = 2.
        self.assertIsNone(prox.strength)

        self.assertEqual(prox.value(coeffs), np.inf)

        out = np.empty(coeffs.shape)
        out.fill(np.mean(coeffs))
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)
        step = 4.
        np.testing.assert_array_almost_equal(prox.call(coeffs, step=step), out)

        coeffs -= 10.
        out.fill(np.mean(coeffs))
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)
        prox.positive = True
        out.fill(0.)
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)

        prox.range = (3, 8)
        prox.positive = False
        coeffs = self.coeffs.copy()
        out = coeffs.copy()
        out[3:8] = np.mean(out[3:8])
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)

        coeffs[3:8] -= 10.
        out[3:8].fill(np.mean(coeffs[3:8]))
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)
        prox.positive = True
        out[3:8].fill(0.)
        np.testing.assert_array_almost_equal(prox.call(coeffs), out)

        self.assertEqual(prox.value(out), 0)
        self.assertEqual(prox.value(coeffs), np.inf)


class ProxEqualityTestFloat32(TestProx, ProxEqualityTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxEqualityTestFloat64(TestProx, ProxEqualityTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
