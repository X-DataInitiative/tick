# License: BSD 3 clause

import unittest

from tick.solver import AGD
from tick.solver.tests import TestSolver

dtype_list = ["float64", "float32"]


class SolverTest(TestSolver):
    def test_solver_agd(self):
        """...Check AGD solver for Logistic Regression with Ridge penalization"""
        solver = AGD(max_iter=100, verbose=False, linesearch=True)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_agd_sparse_and_dense_consistency(self):
        """...SolverTest AGD can run all glm models and is consistent with sparsity"""

        def create_solver():
            return AGD(max_iter=1, verbose=False)

        self._test_solver_sparse_and_dense_consistency(create_solver)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    for dtype in dtype_list:
        suite.addTest(TestSolver.parameterize_main(SolverTest, dtype=dtype))
    unittest.TextTestRunner().run(suite)
