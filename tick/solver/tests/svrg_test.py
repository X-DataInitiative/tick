# License: BSD 3 clause

import unittest
from warnings import catch_warnings, simplefilter
from itertools import product

import numpy as np
from scipy.linalg.special_matrices import toeplitz
from scipy.sparse import csr_matrix

from tick.linear_model import ModelLinReg

from tick.prox import ProxL1, ProxL1w, ProxTV, ProxEquality, \
    ProxElasticNet

from tick.solver import SVRG
from . import TestSolver
from ..build.solver import SVRG as _SVRG


class Test(TestSolver):
    @staticmethod
    def simu_linreg_data(n_samples=5000, n_features=50, interc=-1., p_nnz=0.3):
        np.random.seed(123)
        idx = np.arange(1, n_features + 1)
        weights = (-1) ** (idx - 1) * np.exp(-idx / 10.)
        corr = 0.5
        cov = toeplitz(corr ** np.arange(0, n_features))
        X = np.random.multivariate_normal(np.zeros(n_features), cov,
                                          size=n_samples)
        X *= np.random.binomial(1, p_nnz, size=X.shape)
        idx = np.nonzero(X.sum(axis=1))
        X = X[idx]
        n_samples = X.shape[0]
        noise = np.random.randn(n_samples)
        y = X.dot(weights) + noise
        if interc:
            y += interc
        return X, y

    @staticmethod
    def get_dense_and_sparse_linreg_model(X_dense, y, fit_intercept=True):
        X_sparse = csr_matrix(X_dense)
        model_dense = ModelLinReg(fit_intercept).fit(X_dense, y)
        model_spars = ModelLinReg(fit_intercept).fit(X_sparse, y)
        return model_dense, model_spars

    def test_solver_svrg(self):
        """...Check SVRG solver for a Logistic Regression with Ridge
        penalization
        """
        solver = SVRG(step=1e-3, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=True, model="logreg",
                          decimal=1)

    def test_svrg_sparse_and_dense_consistency(self):
        """...Test SVRG can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SVRG(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_variance_reduction_setting(self):
        """...Test that SVRG variance_reduction parameter behaves correctly
        """
        svrg = SVRG()
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Last)

        svrg = SVRG(variance_reduction='rand')
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'avg'
        self.assertEqual(svrg.variance_reduction, 'avg')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Average)

        svrg.variance_reduction = 'rand'
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Random)

        svrg.variance_reduction = 'last'
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         _SVRG.VarianceReductionMethod_Last)

        msg = '^variance_reduction should be one of "avg, last, rand", ' \
              'got "stuff"$'
        with self.assertRaisesRegex(ValueError, msg):
            svrg = SVRG(variance_reduction='stuff')
        with self.assertRaisesRegex(ValueError, msg):
            svrg.variance_reduction = 'stuff'

        X, y = self.simu_linreg_data()
        model_dense, model_spars = self.get_dense_and_sparse_linreg_model(X, y)
        try:
            svrg.set_model(model_dense)
            svrg.variance_reduction = 'avg'
            svrg.variance_reduction = 'last'
            svrg.variance_reduction = 'rand'
            svrg.set_model(model_spars)
            svrg.variance_reduction = 'last'
            svrg.variance_reduction = 'rand'
        except Exception:
            self.fail('Setting variance_reduction in these cases should have '
                      'been ok')

        msg = "'avg' variance reduction cannot be used with sparse datasets"
        with catch_warnings(record=True) as w:
            simplefilter('always')
            svrg.set_model(model_spars)
            svrg.variance_reduction = 'avg'
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(str(w[0].message), msg)

    def test_step_type_setting(self):
        """...Test that SVRG step_type parameter behaves correctly
        """
        svrg = SVRG()
        self.assertEqual(svrg.step_type, 'fixed')
        self.assertEqual(svrg._solver.get_step_type(),
                         _SVRG.StepType_Fixed)

        svrg = SVRG(step_type='bb')
        self.assertEqual(svrg.step_type, 'bb')
        self.assertEqual(svrg._solver.get_step_type(),
                         _SVRG.StepType_BarzilaiBorwein)

        svrg.step_type = 'fixed'
        self.assertEqual(svrg.step_type, 'fixed')
        self.assertEqual(svrg._solver.get_step_type(),
                         _SVRG.StepType_Fixed)

        svrg.step_type = 'bb'
        self.assertEqual(svrg.step_type, 'bb')
        self.assertEqual(svrg._solver.get_step_type(),
                         _SVRG.StepType_BarzilaiBorwein)

    def test_set_model(self):
        """...Test SVRG set_model
        """
        X, y = self.simu_linreg_data()
        _, model_spars = self.get_dense_and_sparse_linreg_model(X, y)
        svrg = SVRG(variance_reduction='avg')
        msg = "'avg' variance reduction cannot be used with sparse datasets. " \
              "Please change `variance_reduction` before passing sparse data."
        with catch_warnings(record=True) as w:
            simplefilter('always')

            svrg.set_model(model_spars)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertEqual(str(w[0].message), msg)

    def test_dense_and_sparse_match(self):
        """...Test in SVRG that dense and sparse code matches in all possible
        settings
        """
        variance_reductions = ['last', 'rand']
        rand_types = ['perm', 'unif']
        seed = 123
        tol = 0.
        max_iter = 50

        n_samples = 500
        n_features = 20

        # Crazy prox examples
        proxs = [
            ProxTV(strength=1e-2, range=(5, 13), positive=True),
            ProxElasticNet(strength=1e-2, ratio=0.9),
            ProxEquality(range=(0, n_features)),
            ProxL1(strength=1e-3, range=(5, 17)),
            ProxL1w(strength=1e-3, weights=np.arange(5, 17, dtype=np.double),
                    range=(5, 17)),
        ]

        for intercept in [-1, None]:
            X, y = self.simu_linreg_data(interc=intercept,
                                         n_features=n_features,
                                         n_samples=n_samples)

            fit_intercept = intercept is not None
            model_dense, model_spars = self.get_dense_and_sparse_linreg_model(
                    X, y, fit_intercept=fit_intercept)
            step = 1 / model_spars.get_lip_max()

            for variance_reduction, rand_type, prox in product(
                    variance_reductions, rand_types, proxs):
                solver_sparse = SVRG(step=step, tol=tol, max_iter=max_iter,
                                     verbose=False,
                                     variance_reduction=variance_reduction,
                                     rand_type=rand_type, seed=seed) \
                    .set_model(model_spars) \
                    .set_prox(prox)

                solver_dense = SVRG(step=step, tol=tol, max_iter=max_iter,
                                    verbose=False,
                                    variance_reduction=variance_reduction,
                                    rand_type=rand_type, seed=seed) \
                    .set_model(model_dense) \
                    .set_prox(prox)

                solver_sparse.solve()
                solver_dense.solve()
                np.testing.assert_array_almost_equal(solver_sparse.solution,
                                                     solver_dense.solution, 7)



    def test_asvrg_sparse_and_dense_consistency(self):
        """...Test ASVRG can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SVRG(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed, n_threads=2)


if __name__ == '__main__':
    unittest.main()
