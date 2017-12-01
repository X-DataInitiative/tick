# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal
import numpy as np

from tick.prox import ProxL2
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxL2(self):
        """...Test of ProxL2
        """
        coeffs = self.coeffs.copy()
        out = coeffs.copy()
        strength = 3e-2
        step = 1.7

        prox = ProxL2(strength)
        thresh = step * strength * coeffs.shape[0] ** 0.5
        norm = np.linalg.norm(coeffs)
        out *= (1. - thresh / norm) * ((1. - thresh / norm) > 0)
        self.assertAlmostEqual(prox.value(coeffs),
                               strength * coeffs.shape[0] ** 0.5 * norm,
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=step), out, decimal=10)

        start = 3
        end = 8
        range = (start, end)
        prox = ProxL2(strength, range)
        thresh = step * strength * (end - start) ** 0.5
        norm = np.linalg.norm(coeffs[3:8])
        out = coeffs.copy()
        out[3:8] *= (1. - thresh / norm) * ((1. - thresh / norm) > 0)

        self.assertAlmostEqual(prox.value(coeffs),
                               strength * (end - start) ** 0.5 * norm,
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=step), out, decimal=10)

        prox = ProxL2(strength, range, positive=True)
        thresh = step * strength * (end - start) ** 0.5
        norm = np.linalg.norm(coeffs[3:8])
        out = coeffs.copy()
        out[3:8] *= (1. - thresh / norm) * ((1. - thresh / norm) > 0)
        idx = out[3:8] < 0
        out[3:8][idx] = 0

        self.assertAlmostEqual(prox.value(coeffs),
                               strength * (end - start) ** 0.5 * norm,
                               delta=1e-15)
        assert_almost_equal(prox.call(coeffs, step=step), out, decimal=10)


if __name__ == '__main__':
    unittest.main()
