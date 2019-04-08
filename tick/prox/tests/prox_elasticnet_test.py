# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxElasticNet, ProxL1, ProxL2Sq
from tick.prox.tests.prox import TestProx


class ProxElasticNetTest(object):
    def test_ProxElasticNet(self):
        """...Test of ProxElasticNet
        """
        coeffs = self.coeffs.copy().astype(self.dtype)

        l_enet = 3e-2
        ratio = .3
        t = 1.7
        prox_enet = ProxElasticNet(l_enet, ratio=ratio).astype(self.dtype)
        prox_l1 = ProxL1(ratio * l_enet).astype(self.dtype)
        prox_l2 = ProxL2Sq((1 - ratio) * l_enet).astype(self.dtype)

        self.assertAlmostEqual(
            prox_enet.value(coeffs),
            prox_l1.value(coeffs) + prox_l2.value(coeffs), delta=self.delta)

        out = coeffs.copy()
        prox_l1.call(out, t, out)
        prox_l2.call(out, t, out)
        assert_almost_equal(prox_enet.call(coeffs, step=t), out, decimal=10)

        prox_enet = ProxElasticNet(l_enet, ratio=ratio,
                                   positive=True).astype(self.dtype)
        prox_l1 = ProxL1(ratio * l_enet, positive=True).astype(self.dtype)
        prox_l2 = ProxL2Sq((1 - ratio) * l_enet,
                           positive=True).astype(self.dtype)

        self.assertAlmostEqual(
            prox_enet.value(coeffs),
            prox_l1.value(coeffs) + prox_l2.value(coeffs), delta=self.delta)

        out = coeffs.copy()
        prox_l1.call(out, t, out)
        prox_l2.call(out, t, out)
        assert_almost_equal(prox_enet.call(coeffs, step=t), out, decimal=10)


class ProxElasticNetTestFloat32(TestProx, ProxElasticNetTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxElasticNetTestFloat64(TestProx, ProxElasticNetTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
