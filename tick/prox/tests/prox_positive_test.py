# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxPositive
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxPositive(self):
        """...Test of ProxPositive
        """
        coeffs = self.coeffs.copy()

        prox = ProxPositive()
        out = coeffs.copy()
        out[out < 0] = 0
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-15)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)

        prox = ProxPositive((3, 8))
        out = coeffs.copy()
        idx = out[3:8] < 0
        out[3:8][idx] = 0
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-15)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)


if __name__ == '__main__':
    unittest.main()
