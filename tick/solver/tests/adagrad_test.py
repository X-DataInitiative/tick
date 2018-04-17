# License: BSD 3 clause

import unittest

from tick.solver import AdaGrad
from tick.solver.tests import TestSolver

dtype_list = ["float64", "float32"]


class SolverTest(TestSolver):
    def test_solver_adagrad(self):
        """...Check AdaGrad solver for Logistic Regression with Ridge penalization"""
        solver = AdaGrad(max_iter=100, verbose=False, seed=SolverTest.sto_seed,
                         step=0.01)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)


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
