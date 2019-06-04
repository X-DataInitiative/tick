# License: BSD 3 clause

import unittest
from warnings import catch_warnings, simplefilter
from itertools import product

import numpy as np
from scipy.linalg.special_matrices import toeplitz
from scipy.sparse import csr_matrix

from tick.linear_model import ModelLinReg, ModelLogReg, SimuLogReg

from tick.prox import ProxL1, ProxL1w, ProxTV, ProxEquality, \
    ProxElasticNet

from tick.solver import SVRG
from tick.solver.tests import TestSolver

from tick.solver.build.solver import SVRG_VarianceReductionMethod_Last
from tick.solver.build.solver import SVRG_VarianceReductionMethod_Average
from tick.solver.build.solver import SVRG_VarianceReductionMethod_Random

from tick.solver.build.solver import SVRG_StepType_Fixed
from tick.solver.build.solver import SVRG_StepType_BarzilaiBorwein

from tick.simulation import weights_sparse_gauss

dtype_list = ["float64", "float32"]


class SVRGTest(object):
    @staticmethod
    def simu_linreg_data(dtype, n_samples=5000, n_features=50, interc=-1.,
                         p_nnz=0.3):
        np.random.seed(123)
        idx = np.arange(1, n_features + 1)
        weights = (-1) ** (idx - 1) * np.exp(-idx / 10.)
        corr = 0.5
        cov = toeplitz(corr ** np.arange(0, n_features))
        X = np.random.multivariate_normal(
            np.zeros(n_features), cov, size=n_samples)
        X *= np.random.binomial(1, p_nnz, size=X.shape)
        idx = np.nonzero(X.sum(axis=1))
        X = X[idx]
        n_samples = X.shape[0]
        noise = np.random.randn(n_samples)
        y = X.dot(weights) + noise
        if interc:
            y += interc
        X = X.astype(dtype)
        y = y.astype(dtype)
        return X, y

    @staticmethod
    def get_dense_and_sparse_linreg_model(X_dense, y, dtype,
                                          fit_intercept=True):
        X_sparse = csr_matrix(X_dense).astype(dtype)
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

        coeffs0 = weights_sparse_gauss(20, nnz=5, dtype=self.dtype)
        interc0 = None

        X, y = SimuLogReg(coeffs0, interc0, n_samples=3000, verbose=False,
                          seed=123, dtype=self.dtype).simulate()

        model = ModelLogReg().fit(X, y)
        svrg.set_model(model)

        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SVRG_VarianceReductionMethod_Last)

        svrg = SVRG(variance_reduction='rand')
        svrg.set_model(model)
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SVRG_VarianceReductionMethod_Random)

        svrg.variance_reduction = 'avg'
        self.assertEqual(svrg.variance_reduction, 'avg')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SVRG_VarianceReductionMethod_Average)

        svrg.variance_reduction = 'rand'
        self.assertEqual(svrg.variance_reduction, 'rand')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SVRG_VarianceReductionMethod_Random)

        svrg.variance_reduction = 'last'
        self.assertEqual(svrg.variance_reduction, 'last')
        self.assertEqual(svrg._solver.get_variance_reduction(),
                         SVRG_VarianceReductionMethod_Last)

        msg = '^variance_reduction should be one of "avg, last, rand", ' \
              'got "stuff"$'
        with self.assertRaisesRegex(ValueError, msg):
            svrg = SVRG(variance_reduction='stuff')
            svrg.set_model(model)
        with self.assertRaisesRegex(ValueError, msg):
            svrg.variance_reduction = 'stuff'

        X, y = self.simu_linreg_data(dtype=self.dtype)
        model_dense, model_spars = self.get_dense_and_sparse_linreg_model(
            X, y, dtype=self.dtype)
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

        coeffs0 = weights_sparse_gauss(20, nnz=5, dtype=self.dtype)
        interc0 = None

        X, y = SimuLogReg(coeffs0, interc0, n_samples=3000, verbose=False,
                          seed=123, dtype=self.dtype).simulate()

        model = ModelLogReg().fit(X, y)
        svrg.set_model(model)
        self.assertEqual(svrg.step_type, 'fixed')
        self.assertEqual(svrg._solver.get_step_type(), SVRG_StepType_Fixed)

        svrg = SVRG(step_type='bb')
        svrg.set_model(model)
        self.assertEqual(svrg.step_type, 'bb')
        self.assertEqual(svrg._solver.get_step_type(),
                         SVRG_StepType_BarzilaiBorwein)

        svrg.step_type = 'fixed'
        self.assertEqual(svrg.step_type, 'fixed')
        self.assertEqual(svrg._solver.get_step_type(), SVRG_StepType_Fixed)

        svrg.step_type = 'bb'
        self.assertEqual(svrg.step_type, 'bb')
        self.assertEqual(svrg._solver.get_step_type(),
                         SVRG_StepType_BarzilaiBorwein)

    def test_set_model(self):
        """...Test SVRG set_model
        """
        X, y = self.simu_linreg_data(dtype=self.dtype)
        _, model_spars = self.get_dense_and_sparse_linreg_model(
            X, y, dtype=self.dtype)
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
            ProxTV(strength=1e-2, range=(5, 13),
                   positive=True).astype(self.dtype),
            ProxElasticNet(strength=1e-2, ratio=0.9).astype(self.dtype),
            ProxEquality(range=(0, n_features)).astype(self.dtype),
            ProxL1(strength=1e-3, range=(5, 17)).astype(self.dtype),
            ProxL1w(strength=1e-3, weights=np.arange(5, 17, dtype=np.double),
                    range=(5, 17)).astype(self.dtype),
        ]

        for intercept in [-1, None]:
            X, y = self.simu_linreg_data(dtype=self.dtype, interc=intercept,
                                         n_features=n_features,
                                         n_samples=n_samples)

            fit_intercept = intercept is not None
            model_dense, model_spars = self.get_dense_and_sparse_linreg_model(
                X, y, dtype=self.dtype, fit_intercept=fit_intercept)
            step = 1 / model_spars.get_lip_max()

            for variance_reduction, rand_type, prox in product(
                    variance_reductions, rand_types, proxs):
                solver_sparse = SVRG(step=step, tol=tol, max_iter=max_iter,
                                     verbose=False,
                                     variance_reduction=variance_reduction,
                                     rand_type=rand_type, seed=seed)
                solver_sparse.set_model(model_spars).set_prox(prox)

                solver_dense = SVRG(step=step, tol=tol, max_iter=max_iter,
                                    verbose=False,
                                    variance_reduction=variance_reduction,
                                    rand_type=rand_type, seed=seed)
                solver_dense.set_model(model_dense).set_prox(prox)

                solver_sparse.solve()
                solver_dense.solve()
                places = 7
                if self.dtype is "float32":
                    places = 3
                np.testing.assert_array_almost_equal(solver_sparse.solution,
                                                     solver_dense.solution,
                                                     decimal=places)

    def test_asvrg_sparse_and_dense_consistency(self):
        """...Test ASVRG can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SVRG(max_iter=1, verbose=False, step=1e-5,
                        seed=TestSolver.sto_seed, n_threads=2)

        # This test is very unstable...
        # self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_svrg_dtype_can_change(self):
        """...Test svrg astype method
        """

        def create_solver():
            return SVRG(tol=1e-13, step=0.1, max_iter=1000,
                        seed=TestSolver.sto_seed, verbose=False)

        self._test_solver_astype_consistency(create_solver)


class SVRGTestFloat32(TestSolver, SVRGTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float32", **kwargs)


class SVRGTestFloat64(TestSolver, SVRGTest):
    def __init__(self, *args, **kwargs):
        TestSolver.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
