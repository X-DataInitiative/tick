# License: BSD 3 clause

import unittest

from tick.solver import AdaGrad
from tick.solver.tests import TestSolver


class AdagradTest(object):
    def test_solver_adagrad(self):
        """...Check AdaGrad solver for Logistic Regression with Ridge penalization
        """
        solver = AdaGrad(max_iter=100, verbose=False, seed=TestSolver.sto_seed,
                         step=0.01)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_adagrad_dtype_can_change(self):
        """...Test adagrad astype method
        """

        def create_solver():
            return AdaGrad(max_iter=100, verbose=False, step=0.1,
                           seed=TestSolver.sto_seed)

        self._test_solver_astype_consistency(create_solver)


class AdagradTestFloat32(TestSolver, AdagradTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class AdagradTestFloat64(TestSolver, AdagradTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
