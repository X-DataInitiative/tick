# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.prox import ProxL1w
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxL1w(self):
        """...Test of test_ProxL1w
        """
        coeffs = self.coeffs.copy()

        l_l1 = 3e-2
        t = 1.7

        weights = np.arange(coeffs.shape[0], dtype=np.double)

        prox = ProxL1w(l_l1, weights)
        thresh = t * l_l1 * weights
        out = np.sign(coeffs) * (np.abs(coeffs) - thresh) \
              * (np.abs(coeffs) > thresh)
        self.assertAlmostEqual(prox.value(coeffs),
                               l_l1 * (weights * np.abs(coeffs)).sum(),
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)

        weights = np.arange(coeffs.shape[0], dtype=np.double)[3:8]
        prox = ProxL1w(l_l1, weights, (3, 8))
        thresh = t * l_l1 * weights
        sub_coeffs = coeffs[3:8]
        out = coeffs.copy()
        out[3:8] = np.sign(sub_coeffs) \
                   * (np.abs(sub_coeffs) - thresh) \
                   * (np.abs(sub_coeffs) > thresh)
        self.assertAlmostEqual(prox.value(coeffs),
                               l_l1 * (weights * np.abs(coeffs[3:8])).sum(),
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)

        weights = np.arange(coeffs.shape[0], dtype=np.double)[3:8]
        prox = ProxL1w(l_l1, weights, range=(3, 8), positive=True)
        thresh = t * l_l1 * weights
        out = coeffs.copy()
        out[3:8] = np.sign(sub_coeffs) \
                   * (np.abs(sub_coeffs) - thresh) \
                   * (np.abs(sub_coeffs) > thresh)
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        self.assertAlmostEqual(prox.value(coeffs),
                               l_l1 * (weights * np.abs(coeffs[3:8])).sum(),
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=t), out, decimal=10)


if __name__ == '__main__':
    unittest.main()
