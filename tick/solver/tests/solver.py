# License: BSD 3 clause

import unittest

import itertools
import numpy as np
from scipy.linalg import norm
from scipy.sparse import csr_matrix

from tick.linear_model import ModelLogReg, ModelPoisReg, ModelLinReg, \
    SimuLinReg, SimuLogReg, SimuPoisReg
from tick.prox import ProxL2Sq, ProxZero, ProxL1
from tick.solver import SVRG, AGD, SGD, SDCA, GD, BFGS, AdaGrad

from tick.simulation import weights_sparse_gauss


class TestSolver(unittest.TestCase):
    n_features = 20
    n_samples = 3000
    l_l2sq = 1e-6
    sto_seed = 179312

    solvers = [SVRG, AGD, SGD, SDCA, GD, BFGS, AdaGrad]

    def __init__(self, *args, dtype="float64", **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dtype = dtype

    @staticmethod
    def generate_logistic_data(n_features, n_samples, dtype,
                               use_intercept=False):
        """ Function to generate labels features y and X that corresponds
        to w, c
        """
        if n_features <= 5:
            raise ValueError("``n_features`` must be larger than 5")
        np.random.seed(12)
        coeffs0 = weights_sparse_gauss(n_features, nnz=5, dtype=dtype)
        if use_intercept:
            interc0 = 2.
        else:
            interc0 = None
        simu = SimuLogReg(coeffs0, interc0, n_samples=n_samples, verbose=False,
                          dtype=dtype)
        X, y = simu.simulate()
        return y, X, coeffs0, interc0

    def check_solver(self, solver, fit_intercept=True, model='logreg',
                     decimal=1):
        """Check solver instance finds same parameters as scipy BFGS

        Parameters
        ----------
        solver : `Solver`
            Instance of a solver to be tested

        fit_intercept : `bool`, default=True
            Model uses intercept is `True`

        model : 'linreg' | 'logreg' | 'poisreg', default='logreg'
            Name of the model used to test the solver

        decimal : `int`, default=1
            Number of decimals required for the test
        """
        # Set seed for data simulation
        dtype = self.dtype

        if np.dtype(dtype) != np.dtype("float64"):
            return

        np.random.seed(12)
        n_samples = TestSolver.n_samples
        n_features = TestSolver.n_features

        coeffs0 = weights_sparse_gauss(n_features, nnz=5, dtype=dtype)
        if fit_intercept:
            interc0 = 2.
        else:
            interc0 = None

        if model == 'linreg':
            X, y = SimuLinReg(coeffs0, interc0, n_samples=n_samples,
                              verbose=False, seed=123,
                              dtype=self.dtype).simulate()
            model = ModelLinReg(fit_intercept=fit_intercept).fit(X, y)
        elif model == 'logreg':
            X, y = SimuLogReg(coeffs0, interc0, n_samples=n_samples,
                              verbose=False, seed=123,
                              dtype=self.dtype).simulate()
            model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
        elif model == 'poisreg':
            X, y = SimuPoisReg(coeffs0, interc0, n_samples=n_samples,
                               verbose=False, seed=123,
                               dtype=self.dtype).simulate()
            # Rescale features to avoid overflows in Poisson simulations
            X /= np.linalg.norm(X, axis=1).reshape(n_samples, 1)
            model = ModelPoisReg(fit_intercept=fit_intercept).fit(X, y)
        else:
            raise ValueError("``model`` must be either 'linreg', 'logreg' or"
                             " 'poisreg'")

        solver.set_model(model)
        strength = 1e-2
        prox = ProxL2Sq(strength, (0, model.n_features))

        if type(solver) is not SDCA:
            solver.set_prox(prox)
        else:
            solver.set_prox(ProxZero().astype(self.dtype))
            solver.l_l2sq = strength

        coeffs_solver = solver.solve()
        # Compare with BFGS
        bfgs = BFGS(max_iter=100,
                    verbose=False).set_model(model).set_prox(prox)
        coeffs_bfgs = bfgs.solve()

        np.testing.assert_almost_equal(coeffs_solver, coeffs_bfgs,
                                       decimal=decimal)

        # We ensure that reached coeffs are not equal to zero
        self.assertGreater(norm(coeffs_solver), 0)

        self.assertAlmostEqual(
            solver.objective(coeffs_bfgs), solver.objective(coeffs_solver),
            delta=1e-2)

    @staticmethod
    def prepare_solver(solver, X, y, fit_intercept=True, model="logistic",
                       prox="l2"):
        if model == "logistic":
            model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
        elif model == "poisson":
            model = ModelPoisReg(fit_intercept=fit_intercept).fit(X, y)
        solver.set_model(model)
        if prox == "l2":
            l_l2sq = TestSolver.l_l2sq
            prox = ProxL2Sq(l_l2sq, (0, model.n_coeffs))
        if prox is not None:
            solver.set_prox(prox)

    def _test_solver_sparse_and_dense_consistency(
            self, create_solver, model_classes=list(
                [ModelLinReg, ModelLogReg, ModelPoisReg]), proxs_classes=list(
                    [ProxL2Sq, ProxL1]), fit_intercepts=list([False, True])):
        """...Test that solvers can run all glm models and are consistent
        with sparsity
        """
        dtype = self.dtype
        n_samples = 50
        n_features = 10
        coeffs0 = weights_sparse_gauss(n_features, nnz=5)
        interc0 = 2.
        seed = 123
        prox_strength = 1e-3

        model_simu_map = {
            ModelLinReg: SimuLinReg,
            ModelPoisReg: SimuPoisReg,
            ModelLogReg: SimuLogReg,
        }

        cases = itertools.product(model_classes, proxs_classes, fit_intercepts)

        for Model, Prox, fit_intercept in cases:

            if fit_intercept:
                interc = interc0
            else:
                interc = None

            Simu = model_simu_map[Model]
            simu = Simu(coeffs0, interc, n_samples=n_samples, seed=seed,
                        verbose=False, dtype=self.dtype)
            X, y = simu.simulate()
            if X.dtype != y.dtype:
                raise ValueError(
                    "Simulation error, features and label dtypes differ")
            X_sparse = csr_matrix(X).astype(self.dtype)

            for sparse in [True, False]:
                model = Model(fit_intercept=fit_intercept)

                if sparse:
                    model.fit(X_sparse, y)
                else:
                    model.fit(X, y)

                prox = Prox(prox_strength, (0, n_features)).astype(self.dtype)
                solver = create_solver()
                solver.set_model(model).set_prox(prox)

                if sparse:
                    iterate_sparse = solver.solve()
                else:
                    iterate_dense = solver.solve()

            error_msg = 'Failed for %s and %s solved with %s' % (
                model.name, prox.name, solver.name)

            if fit_intercept:
                error_msg += ' with intercept'
            else:
                error_msg += ' without intercept'

            self.assertEqual(np.isfinite(iterate_dense).all(), True, error_msg)

            places = 7
            if self.dtype is "float32" or self.dtype is np.dtype("float32"):
                places = 4
            np.testing.assert_almost_equal(iterate_dense, iterate_sparse,
                                           err_msg=error_msg, decimal=places)

    def _test_solver_astype_consistency(self, create_solver):
        # Launch this test only once
        if self.dtype != 'float64':
            return

        prox = ProxL2Sq(0.1)

        use_intercept = True
        y_64, X_64, coeffs0_64, interc0 = self.generate_logistic_data(
            100, 30, 'float64', use_intercept)

        model_64 = ModelLogReg(fit_intercept=use_intercept)
        model_64.fit(X_64, y_64)
        solver_64 = create_solver()
        solver_64.set_model(model_64).set_prox(prox)
        solution_64 = solver_64.solve()

        solver_32 = solver_64.astype('float32')
        solution_32 = solver_32.solve()

        self.assertEqual(solution_64.dtype, 'float64')
        self.assertEqual(solution_32.dtype, 'float32')

        np.testing.assert_array_almost_equal(solution_32, solution_64,
                                             decimal=3)

    def test_set_model_and_set_prox(self):
        np.random.seed(12)
        n_samples = TestSolver.n_samples
        n_features = TestSolver.n_features
        weights0 = weights_sparse_gauss(n_features, nnz=5)
        interc0 = 2.
        model = ModelLinReg()
        msg = '^Passed object ModelLinReg has not been fitted. You must call' \
              ' ``fit`` on it before passing it to ``set_model``$'
        with self.assertRaisesRegex(ValueError, msg):
            for solver_class in self.solvers:
                if solver_class is SDCA:
                    solver = solver_class(l_l2sq=1e-1)
                else:
                    solver = solver_class()
                solver.set_model(model)

        X, y = SimuLinReg(weights0, interc0, n_samples=n_samples,
                          verbose=False, seed=123,
                          dtype=self.dtype).simulate()
        prox = ProxL2Sq(strength=1e-1)
        msg = '^Passed object of class ProxL2Sq is not a Model class$'
        with self.assertRaisesRegex(ValueError, msg):
            for solver_class in self.solvers:
                if solver_class is SDCA:
                    solver = solver_class(l_l2sq=1e-1)
                else:
                    solver = solver_class()
                solver.set_model(prox)
        model.fit(X, y)
        msg = '^Passed object of class ModelLinReg is not a Prox class$'
        with self.assertRaisesRegex(ValueError, msg):
            for solver_class in self.solvers:
                if solver_class is SDCA:
                    solver = solver_class(l_l2sq=1e-1)
                else:
                    solver = solver_class()
                solver.set_model(model).set_prox(model)

    @staticmethod
    def evaluate_model(coeffs, w, c=None):
        if c is None:
            err = norm(coeffs - w)
        else:
            err = abs(c - coeffs[-1])
            err += norm(coeffs[:-1] - w)
        return err
