# License: BSD 3 clause

import unittest

from tick.solver import GD
from tick.solver.tests import TestSolver


class GDTest(object):
    def test_solver_gd(self):
        """...Check GD solver for Logistic Regression with Ridge penalization
        """
        solver = GD(max_iter=100, verbose=False, linesearch=True)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_gd_sparse_and_dense_consistency(self):
        """...Test GD can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return GD(max_iter=1, verbose=False)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_gd_dtype_can_change(self):
        """...Test gd astype method
        """

        def create_solver():
            return GD(max_iter=100, verbose=False, step=0.1)

        self._test_solver_astype_consistency(create_solver)


class GDTestFloat32(TestSolver, GDTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class GDTestFloat64(TestSolver, GDTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
