# License: BSD 3 clause

import unittest

import numpy as np

from tick.prox import ProxElasticNet, ProxL2Sq, ProxL1
from tick.solver import GFB, AGD
from tick.solver.tests import TestSolver

dtype_list = ["float64", "float32"]


class SolverTest(TestSolver):

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
        y, X, w, c = SolverTest.generate_logistic_data(
            n_features=n_features, n_samples=n_samples, dtype=self.dtype)
        strength = 1e-3
        ratio = 0.3
        prox_elasticnet = ProxElasticNet(strength, ratio)
        prox_l1 = ProxL1(strength * ratio)
        prox_l2 = ProxL2Sq(strength * (1 - ratio))

        # First we get GFB solution with prox l1 and prox l2
        gfb = GFB(tol=1e-13, max_iter=1000, verbose=False, step=1)
        SolverTest.prepare_solver(gfb, X, y, prox=None)
        gfb.set_prox([prox_l1, prox_l2])
        gfb_solution = gfb.solve()

        # Then we get AGD solution with prox ElasticNet
        agd = AGD(
            tol=1e-13, max_iter=1000, verbose=False, step=0.5, linesearch=False)
        SolverTest.prepare_solver(agd, X, y, prox=prox_elasticnet)
        agd_solution = agd.solve()

        # Finally we assert that both algorithms lead to the same solution
        np.testing.assert_almost_equal(gfb_solution, agd_solution, decimal=1)


def parameterize(klass, dtype):
    testnames = unittest.TestLoader().getTestCaseNames(klass)
    suite = unittest.TestSuite()
    for name in testnames:
        suite.addTest(klass(name, dtype=dtype))
    return suite


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for dt in dtype_list:
        suite.addTest(parameterize(SolverTest, dtype=dt))
    unittest.TextTestRunner().run(suite)
