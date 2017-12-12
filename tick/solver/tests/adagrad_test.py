# License: BSD 3 clause

import unittest

from tick.solver import AdaGrad
from . import TestSolver


class Test(TestSolver):
    def test_solver_adagrad(self):
        """...Check AdaGrad solver for Logistic Regression with Ridge
        penalization
        """
        solver = AdaGrad(max_iter=100, verbose=False, seed=Test.sto_seed,
                         step=0.01)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

if __name__ == '__main__':
    unittest.main()
