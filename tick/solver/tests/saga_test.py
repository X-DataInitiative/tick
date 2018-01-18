# License: BSD 3 clause

import unittest

from tick.solver import SAGA
from . import TestSolver
from ..build.solver import SAGA as _SAGA

from tick.survival import SimuCoxReg, ModelCoxRegPartialLik
from tick.simulation import weights_sparse_gauss


class Test(TestSolver):
    def test_solver_saga(self):
        """...Check SAGA solver for a Logistic Regression with Ridge
        penalization
        """
        solver = SAGA(step=1e-3, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_saga_sparse_and_dense_consistency(self):
        """...Test SAGA can run all glm models and is consistent with sparsity
        """
        def create_solver():
            return SAGA(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_variance_reduction_setting(self):
        """...Test SAGA variance_reduction parameter is correctly set
        """
        svrg = SAGA()
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SAGA.VarianceReductionMethod_Last)

        svrg = SAGA(variance_reduction='rand')
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SAGA.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'avg'
        self.assertEqual(svrg.variance_reduction, 'avg')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SAGA.VarianceReductionMethod_Average)

        svrg.variance_reduction = 'rand'
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SAGA.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'last'
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SAGA.VarianceReductionMethod_Last)

        with self.assertRaises(ValueError):
            svrg.variance_reduction = 'wrong_name'

    def test_set_model(self):
        """...Test set_model of saga, should only accept childs of
        ModelGeneralizedLinear"""
        # We try to pass a ModelCoxRegPartialLik which is not a generalized
        # linear model to SAGA to check that the error is raised
        msg = '^SAGA accepts only childs of `ModelGeneralizedLinear`$'
        with self.assertRaisesRegex(ValueError, msg):
            w = weights_sparse_gauss(n_weights=2, nnz=0)
            X, T, C = SimuCoxReg(w).simulate()
            model = ModelCoxRegPartialLik().fit(X, T, C)
            SAGA().set_model(model)

        msg = '^SAGA accepts only childs of `ModelGeneralizedLinear`$'
        with self.assertRaisesRegex(RuntimeError, msg):
            w = weights_sparse_gauss(n_weights=2, nnz=0)
            X, T, C = SimuCoxReg(w).simulate()
            model = ModelCoxRegPartialLik().fit(X, T, C)
            saga = SAGA()
            saga._solver.set_model(model._model)


if __name__ == '__main__':
    unittest.main()
