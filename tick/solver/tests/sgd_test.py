# License: BSD 3 clause

import unittest

from tick.solver import SGD
from . import TestSolver


class Test(TestSolver):
    def test_solver_sgd(self):
        """...Check SGD solver for Logistic Regression with Ridge
        penalization
        """
        solver = SGD(max_iter=100, verbose=False, seed=Test.sto_seed,
                     step=200)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=0)

    def test_sgd_sparse_and_dense_consistency(self):
        """...Test SGD can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SGD(max_iter=1, verbose=False, step=1e-5,
                       seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)


if __name__ == '__main__':
    unittest.main()
