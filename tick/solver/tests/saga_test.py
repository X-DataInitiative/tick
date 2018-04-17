# License: BSD 3 clause

import unittest

import numpy as np

from tick.solver import SAGA

from tick.solver.tests import TestSolver
from tick.solver.build.solver import SAGADouble as _SAGA

from tick.linear_model import ModelLogReg, SimuLogReg

from tick.survival import SimuCoxReg, ModelCoxRegPartialLik

from tick.solver.build.solver import SAGA_VarianceReductionMethod_Last
from tick.solver.build.solver import SAGA_VarianceReductionMethod_Average
from tick.solver.build.solver import SAGA_VarianceReductionMethod_Random

from tick.simulation import weights_sparse_gauss

dtype_list = ["float64", "float32"]


class SolverTest(TestSolver):
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

    def test_variance_reduction_setting(self):
        """...SolverTest SAGA variance_reduction parameter is correctly set"""
        svrg = SAGA()

        coeffs0 = weights_sparse_gauss(20, nnz=5, dtype=self.dtype)
        interc0 = None

        X, y = SimuLogReg(coeffs0, interc0, n_samples=3000, verbose=False,
                          seed=123, dtype=self.dtype).simulate()

        model = ModelLogReg().fit(X, y)
        svrg.set_model(model)
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SAGA_VarianceReductionMethod_Last)

        svrg = SAGA(variance_reduction='rand')
        svrg.set_model(model)
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SAGA_VarianceReductionMethod_Random)

        svrg.variance_reduction = 'avg'
        self.assertEqual(svrg.variance_reduction, 'avg')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SAGA_VarianceReductionMethod_Average)

        svrg.variance_reduction = 'rand'
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SAGA_VarianceReductionMethod_Random)

        svrg.variance_reduction = 'last'
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SAGA_VarianceReductionMethod_Last)

        with self.assertRaises(ValueError):
            svrg.variance_reduction = 'wrong_name'

    def test_set_model(self):
        """...SolverTest set_model of saga, should only accept childs of
      ModelGeneralizedLinear"""
        # We try to pass a ModelCoxRegPartialLik which is not a generalized
        # linear model to SAGA to check that the error is raised
        msg = '^SAGA accepts only childs of `ModelGeneralizedLinear`$'
        with self.assertRaisesRegex(ValueError, msg):
            w = weights_sparse_gauss(n_weights=2, nnz=0, dtype=self.dtype)
            X, T, C = SimuCoxReg(w, dtype=self.dtype).simulate()
            model = ModelCoxRegPartialLik().fit(X, T, C)
            SAGA().set_model(model)


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
