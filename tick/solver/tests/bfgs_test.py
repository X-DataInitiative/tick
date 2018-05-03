# License: BSD 3 clause

import unittest

import numpy as np

from tick.linear_model import ModelLogReg
from tick.prox import ProxL2Sq
from tick.solver import BFGS
from tick.solver.tests import TestSolver
from tick.linear_model import SimuLogReg
from tick.simulation import weights_sparse_gauss


class Test(TestSolver):
    def test_solver_bfgs(self):
        """...Check BFGS solver for Logistic Regression with Ridge
    penalization
    """
        # It is the reference solver used in other unittests so we check that
        # it's actually close to the true parameter of the simulated dataset
        np.random.seed(12)
        n_samples = 3000
        n_features = 10
        coeffs0 = weights_sparse_gauss(n_features, nnz=5).astype(self.dtype)
        interc0 = 2.
        X, y = SimuLogReg(coeffs0, interc0, n_samples=n_samples, verbose=False,
                          dtype=self.dtype).simulate()
        model = ModelLogReg(fit_intercept=True).fit(X, y)
        prox = ProxL2Sq(strength=1e-6)
        solver = BFGS(max_iter=100, print_every=1, verbose=False,
                      tol=1e-6).set_model(model).set_prox(prox)
        coeffs = solver.solve()
        err = TestSolver.evaluate_model(coeffs, coeffs0, interc0)
        self.assertAlmostEqual(err, 0., delta=5e-1)


if __name__ == '__main__':
    unittest.main()
