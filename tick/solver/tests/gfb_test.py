# License: BSD 3 clause

import unittest

import numpy as np

from tick.prox import ProxElasticNet, ProxL2Sq, ProxL1
from tick.solver import GFB, AGD
from tick.solver.tests import TestSolver


class GDBTest(object):
    def test_solver_gfb(self):
        """...Check GFB's solver for a Logistic Regression with ElasticNet
        penalization

        Notes
        -----
        Using GFB solver with l1 and l2 penalizations is obviously a bad
        idea as ElasticNet prox is meant to do this, but it allows us to
        compare with another algorithm.
        """
        n_samples = 200
        n_features = 10
        y, X, w, c = TestSolver.generate_logistic_data(
            n_features=n_features, n_samples=n_samples, dtype=self.dtype)
        strength = 1e-3
        ratio = 0.3
        prox_elasticnet = ProxElasticNet(strength, ratio).astype(self.dtype)
        prox_l1 = ProxL1(strength * ratio).astype(self.dtype)
        prox_l2 = ProxL2Sq(strength * (1 - ratio)).astype(self.dtype)

        # First we get GFB solution with prox l1 and prox l2
        gfb = GFB(tol=1e-13, max_iter=1000, verbose=False, step=1)
        TestSolver.prepare_solver(gfb, X, y, prox=None)
        gfb.set_prox([prox_l1, prox_l2])
        gfb_solution = gfb.solve()

        # Then we get AGD solution with prox ElasticNet
        agd = AGD(tol=1e-13, max_iter=1000, verbose=False, step=0.5,
                  linesearch=False)
        TestSolver.prepare_solver(agd, X, y, prox=prox_elasticnet)
        agd_solution = agd.solve()

        # Finally we assert that both algorithms lead to the same solution
        np.testing.assert_almost_equal(gfb_solution, agd_solution, decimal=1)

    def test_gfb_dtype_can_change(self):
        """...Test gfb astype method
        """

        def create_solver():
            return GFB(max_iter=100, verbose=False, step=0.1)

        self._test_solver_astype_consistency(create_solver)


class GDBTestFloat32(TestSolver, GDBTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class GDBTestFloat64(TestSolver, GDBTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
