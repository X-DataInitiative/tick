# License: BSD 3 clause

import unittest

from tick.solver import GD
from tick.solver.tests import TestSolver

dtype_list = ["float32"]


class SolverTest(TestSolver):

    def test_solver_gd(self):
        """...Check GD solver for Logistic Regression with Ridge penalization"""
        solver = GD(max_iter=100, verbose=False, linesearch=True)
        self.check_solver(solver, fit_intercept=True, model="logreg", decimal=1)

    def test_gd_sparse_and_dense_consistency(self):
        """...SolverTest GD can run all glm models and is consistent with sparsity"""

        def create_solver():
            return GD(max_iter=1, verbose=False)

        self._test_solver_sparse_and_dense_consistency(create_solver)


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
