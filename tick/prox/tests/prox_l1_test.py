# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.prox import ProxL1
from tick.prox.tests.prox import TestProx


class ProxL1Test(object):
    def test_ProxL1(self):
        """...Test of ProxL1
        """
        coeffs = self.coeffs.copy().astype(self.dtype)

        l_l1 = 3e-2
        t = 1.7

        prox = ProxL1(l_l1).astype(self.dtype)
        thresh = t * l_l1
        out = np.sign(coeffs) * (np.abs(coeffs) - thresh) \
              * (np.abs(coeffs) > thresh)
        self.assertAlmostEqual(
            prox.value(coeffs),
            l_l1 * np.abs(coeffs).sum(), delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)

        prox = ProxL1(l_l1, (3, 8)).astype(self.dtype)
        thresh = t * l_l1
        sub_coeffs = coeffs[3:8]
        out = coeffs.copy()
        out[3:8] = np.sign(sub_coeffs) \
                   * (np.abs(sub_coeffs) - thresh) \
                   * (np.abs(sub_coeffs) > thresh)
        self.assertAlmostEqual(
            prox.value(coeffs),
            l_l1 * np.abs(coeffs[3:8]).sum(), delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)

        prox = ProxL1(l_l1, (3, 8), positive=True).astype(self.dtype)
        thresh = t * l_l1
        sub_coeffs = coeffs[3:8]
        out = coeffs.copy()
        out[3:8] = np.sign(sub_coeffs) * (np.abs(sub_coeffs) - thresh) \
                   * (np.abs(sub_coeffs) > thresh)
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        self.assertAlmostEqual(
            prox.value(coeffs),
            l_l1 * np.abs(coeffs[3:8]).sum(), delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)


class ProxL1TestFloat32(TestProx, ProxL1Test):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxL1TestFloat64(TestProx, ProxL1Test):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
