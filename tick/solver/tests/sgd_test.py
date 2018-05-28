# License: BSD 3 clause

import unittest

from tick.solver import SGD
from tick.solver.tests import TestSolver


class SGDTest(object):
    def test_solver_sgd(self):
        """...Check SGD solver for Logistic Regression with Ridge
        penalization
        """
        solver = SGD(max_iter=100, verbose=False, seed=TestSolver.sto_seed,
                     step=200)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=0)

    def test_sgd_sparse_and_dense_consistency(self):
        """...SGDTest SGD can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SGD(max_iter=1, verbose=False, step=1e-5,
                       seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_sgd_dtype_can_change(self):
        """...Test sgd astype method
        """

        def create_solver():
            return SGD(max_iter=100, verbose=False, step=1e-1,
                       seed=TestSolver.sto_seed)

        self._test_solver_astype_consistency(create_solver)


class SGDTestFloat32(TestSolver, SGDTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class SGDTestFloat64(TestSolver, SGDTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
