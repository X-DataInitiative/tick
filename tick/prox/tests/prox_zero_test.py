# License: BSD 3 clause

import unittest

from numpy.testing import assert_almost_equal

from tick.prox import ProxZero
from tick.prox.tests.prox import TestProx


class Test(TestProx):
    def test_ProxZero(self):
        """...Test of ProxZero
        """
        coeffs = self.coeffs.copy()
        out = coeffs.copy()

        prox = ProxZero()
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-14)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)

        prox = ProxZero((3, 8))
        self.assertAlmostEqual(prox.value(coeffs), 0., delta=1e-14)
        assert_almost_equal(prox.call(coeffs), out, decimal=10)


if __name__ == '__main__':
    unittest.main()
