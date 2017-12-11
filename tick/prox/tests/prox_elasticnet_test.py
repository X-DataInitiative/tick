# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxElasticNet, ProxL1, ProxL2Sq
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxElasticNet(self):
        """...Test of ProxElasticNet
        """
        coeffs = self.coeffs.copy()

        l_enet = 3e-2
        ratio = .3
        t = 1.7
        prox_enet = ProxElasticNet(l_enet, ratio=ratio)
        prox_l1 = ProxL1(ratio * l_enet)
        prox_l2 = ProxL2Sq((1 - ratio) * l_enet)

        self.assertAlmostEqual(prox_enet.value(coeffs),
                               prox_l1.value(coeffs) + prox_l2.value(coeffs),
                               delta=1e-15)

        out = coeffs.copy()
        prox_l1.call(out, t, out)
        prox_l2.call(out, t, out)
        assert_almost_equal(prox_enet.call(coeffs, step=t), out, decimal=10)

        prox_enet = ProxElasticNet(l_enet, ratio=ratio, positive=True)
        prox_l1 = ProxL1(ratio * l_enet, positive=True)
        prox_l2 = ProxL2Sq((1 - ratio) * l_enet, positive=True)

        self.assertAlmostEqual(prox_enet.value(coeffs),
                               prox_l1.value(coeffs) + prox_l2.value(coeffs),
                               delta=1e-15)

        out = coeffs.copy()
        prox_l1.call(out, t, out)
        prox_l2.call(out, t, out)
        assert_almost_equal(prox_enet.call(coeffs, step=t), out, decimal=10)


if __name__ == '__main__':
    unittest.main()
