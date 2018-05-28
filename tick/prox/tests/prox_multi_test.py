# License: BSD 3 clause

import unittest

import numpy as np

from tick.prox import ProxMulti, ProxTV, ProxElasticNet
from tick.prox.tests.prox import TestProx


class ProxMultiTest(object):
    def test_prox_multi(self):
        """...Test of ProxMulti
        """
        coeffs = self.coeffs.copy().astype(self.dtype)
        double_coeffs = np.concatenate([coeffs, coeffs]).astype(self.dtype)
        half_size = coeffs.shape[0]
        full_size = double_coeffs.shape[0]

        l_tv = 0.5
        t = 1.7
        prox_tv = ProxTV(strength=l_tv).astype(self.dtype)
        prox_tv_multi = ProxTV(strength=l_tv,
                               range=(0, half_size)).astype(self.dtype)

        l_enet = 3e-2
        ratio = .3
        prox_enet = ProxElasticNet(l_enet, ratio=ratio).astype(self.dtype)
        prox_enet_multi = ProxElasticNet(l_enet, ratio=ratio,
                                         range=(half_size,
                                                full_size)).astype(self.dtype)

        prox_multi = ProxMulti((prox_tv_multi, prox_enet_multi))

        # Test that the value of the prox is correct
        val_multi = prox_multi.value(double_coeffs)
        val_correct = prox_enet.value(coeffs) + prox_tv.value(coeffs)
        self.assertAlmostEqual(val_multi, val_correct,
                               places=self.decimal_places)

        # Test that the prox is correct
        out1 = prox_tv.call(coeffs, step=t)
        out2 = prox_enet.call(coeffs, step=t)
        out_correct = np.concatenate([out1, out2])
        out_multi = prox_multi.call(double_coeffs, step=t)
        np.testing.assert_almost_equal(out_multi, out_correct)

        # An example with overlapping coefficients
        start1 = 5
        end1 = 13
        start2 = 10
        end2 = 17
        prox_tv = ProxTV(strength=l_tv, range=(start1,
                                               end1)).astype(self.dtype)
        prox_enet = ProxElasticNet(strength=l_enet, ratio=ratio,
                                   range=(start2, end2)).astype(self.dtype)
        prox_multi = ProxMulti((prox_tv, prox_enet))

        val_correct = prox_tv.value(double_coeffs)
        val_correct += prox_enet.value(double_coeffs)
        val_multi = prox_multi.value(double_coeffs)
        self.assertAlmostEqual(val_multi, val_correct)

        out_correct = prox_tv.call(double_coeffs)
        out_correct = prox_enet.call(out_correct)
        out_multi = prox_multi.call(double_coeffs)
        np.testing.assert_almost_equal(out_multi, out_correct)


class ProxMultiTestFloat32(TestProx, ProxMultiTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxMultiTestFloat64(TestProx, ProxMultiTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
