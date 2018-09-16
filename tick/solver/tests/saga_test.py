# License: BSD 3 clause

import unittest
import numpy as np
from tick.solver import SAGA
from tick.solver.tests import TestSolver
from tick.solver.build.solver import SAGADouble as _SAGA
from tick.linear_model import ModelLogReg, SimuLogReg
from tick.survival import SimuCoxReg, ModelCoxRegPartialLik
from tick.simulation import weights_sparse_gauss

class SAGATest(object):
    def test_solver_saga(self):
        """...Check SAGA solver for a Logistic Regression with Ridge penalization"""
        solver = SAGA(step=1e-3, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_saga_sparse_and_dense_consistency(self):
        """...SolverTest SAGA can run all glm models and is consistent with sparsity"""
        def create_solver():
            return SAGA(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_set_model(self):
        """...SolverTest set_model of saga, should only accept childs of
           ModelGeneralizedLinear"""
        # We try to pass a ModelCoxRegPartialLik which is not a generalized
        # linear model to SAGA to check that the error is raised
        msg = '^SAGA accepts only childs of `ModelGeneralizedLinear`$'
        with self.assertRaisesRegex(ValueError, msg):
            w = weights_sparse_gauss(n_weights=2, nnz=0, dtype=self.dtype)
            X, T, C = SimuCoxReg(w, dtype=self.dtype, verbose=False).simulate()
            model = ModelCoxRegPartialLik().fit(X, T, C)
            SAGA().set_model(model)

    def test_saga_dtype_can_change(self):
        """...Test saga astype method
        """
        def create_solver():
            return SAGA(max_iter=100, verbose=False, step=0.01,
                        seed=TestSolver.sto_seed)

        self._test_solver_astype_consistency(create_solver)

class SAGATestFloat32(TestSolver, SAGATest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class SAGATestFloat64(TestSolver, SAGATest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)

if __name__ == '__main__':
    unittest.main()
