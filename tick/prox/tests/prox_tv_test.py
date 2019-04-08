# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.prox import ProxTV
from tick.prox.tests.prox import TestProx


class ProxTVTest(object):
    def test_ProxTV(self):
        """...Test of ProxTV
        """
        coeffs = self.coeffs.copy().astype(self.dtype)
        l_tv = 0.5
        t = 1.7
        out = np.array([
            -0.40102846, -0.40102846, -0.40102846, -0.31364696, -0.31364696,
            1.03937619, 1.03937619, 1.03937619, -0.21598253, -0.21598253
        ])
        prox = ProxTV(l_tv).astype(self.dtype)
        val = l_tv * np.abs(coeffs[1:] - coeffs[:-1]).sum()
        self.assertAlmostEqual(prox.value(coeffs), val, delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)

        coeffs = self.coeffs.copy().astype(self.dtype)
        out = np.array([
            -0.86017247, -0.58127151, -0.6116414, 0.11135304, 0.11135304,
            1.32270952, 1.32270952, 1.32270952, -0.27576309, -1.00620197
        ])
        prox = ProxTV(l_tv, (3, 8)).astype(self.dtype)
        val = l_tv * np.abs(coeffs[3:8][1:] - coeffs[3:8][:-1]).sum()
        self.assertAlmostEqual(prox.value(coeffs), val, delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)

        coeffs = self.coeffs.copy().astype(self.dtype)
        out = np.array([
            -0.86017247, -0.58127151, -0.6116414, 0.11135304, 0.11135304,
            1.32270952, 1.32270952, 1.32270952, -0.27576309, -1.00620197
        ])
        prox = ProxTV(l_tv, (3, 8), positive=True).astype(self.dtype)
        prox.call(coeffs, step=t)
        val = l_tv * np.abs(coeffs[3:8][1:] - coeffs[3:8][:-1]).sum()
        self.assertAlmostEqual(prox.value(coeffs), val, delta=self.delta)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=7)


class ProxTVTestFloat32(TestProx, ProxTVTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxTVTestFloat64(TestProx, ProxTVTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
