# License: BSD 3 clause

import itertools
import unittest
from itertools import product

import numpy as np
from scipy.linalg import norm

from tick.base.inference import InferenceTest
from tick.linear_model import SimuLinReg, LinearRegression
from tick.simulation import weights_sparse_gauss


class Test(InferenceTest):
    def setUp(self):
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.X = np.zeros((5, 5))
        self.y = np.zeros(5)
        self.y[0] = 1

    @staticmethod
    def get_train_data(n_samples=2000, n_features=20, fit_intercept=True):
        np.random.seed(12)
        weights0 = weights_sparse_gauss(n_features)
        if fit_intercept:
            intercept0 = -1.
        else:
            intercept0 = None
        X, y = SimuLinReg(weights0, intercept0, n_samples=n_samples,
                          verbose=False).simulate()
        return X, y, weights0, intercept0

    def test_LinearRegression_fit(self):
        """...Test LinearRegression fit with different solvers and penalties
        """
        fit_intercepts = [False, True]
        n_samples = 2000
        n_features = 20
        X, y, weights0, _ = self.get_train_data(
            n_samples=n_samples, n_features=n_features, fit_intercept=False)
        intercept0 = -1
        for i, (solver, penalty, fit_intercept) \
                in enumerate(product(LinearRegression._solvers.keys(),
                                     LinearRegression._penalties.keys(),
                                     fit_intercepts)):
            if fit_intercept:
                y_ = y + intercept0
            else:
                y_ = y.copy()
            if penalty == 'binarsity':
                learner = LinearRegression(
                    verbose=False, fit_intercept=fit_intercept, solver=solver,
                    penalty=penalty, tol=1e-10, blocks_start=[0],
                    blocks_length=[1])
            else:
                learner = LinearRegression(
                    verbose=False, fit_intercept=fit_intercept, solver=solver,
                    penalty=penalty, tol=1e-10)
            learner.fit(X, y_)
            err = norm(learner.weights - weights0) / n_features ** 0.5
            self.assertLess(err, 3e-2)
            if fit_intercept:
                self.assertLess(abs(learner.intercept - intercept0), 3e-2)

    def test_LinearRegression_warm_start(self):
        """...Test LinearRegression warm start
        """
        X, y, weights0, intercept0 = Test.get_train_data()

        fit_intercepts = [True, False]
        cases = itertools.product(LinearRegression._solvers.keys(),
                                  fit_intercepts)

        for solver, fit_intercept in cases:
            solver_kwargs = {
                'solver': solver,
                'max_iter': 2,
                'fit_intercept': fit_intercept,
                'warm_start': True,
                'tol': 0
            }

            learner = LinearRegression(**solver_kwargs)
            learner.fit(X, y)
            if fit_intercept:
                coeffs_1 = np.hstack((learner.weights, learner.intercept))
            else:
                coeffs_1 = learner.weights

            learner.fit(X, y)
            if fit_intercept:
                coeffs_2 = np.hstack((learner.weights, learner.intercept))
            else:
                coeffs_2 = learner.weights
            # Thanks to warm start objective should have decreased
            self.assertLess(
                learner._solver_obj.objective(coeffs_2),
                learner._solver_obj.objective(coeffs_1))

    @staticmethod
    def specific_solver_kwargs(solver):
        """...A simple method to as systematically some kwargs to our tests
        """
        return dict()

    def test_LinearRegression_settings(self):
        """...Test LinearRegression basic settings
        """
        # solver
        from tick.solver import AGD, GD, BFGS, SGD, SVRG, SDCA
        solvers = {
            'AGD': AGD,
            'BFGS': BFGS,
            'GD': GD,
            'SGD': SGD,
            'SVRG': SVRG,
            'SDCA': SDCA
        }
        solver_class_map = LinearRegression._solvers
        for solver in LinearRegression._solvers.keys():
            learner = LinearRegression(solver=solver,
                                       **Test.specific_solver_kwargs(solver))
            solver_class = solvers[solver_class_map[solver]]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

        msg = '^``solver`` must be one of agd, gd, svrg, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            LinearRegression(solver='wrong_name')

        # prox
        prox_class_map = LinearRegression._penalties
        for penalty in LinearRegression._penalties.keys():
            if penalty == 'binarsity':
                learner = LinearRegression(penalty=penalty, blocks_start=[0],
                                           blocks_length=[1])
            else:
                learner = LinearRegression(penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

        msg = '^``penalty`` must be one of binarsity, elasticnet, l1, l2, ' \
              'none, tv, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            LinearRegression(penalty='wrong_name')

    def test_LinearRegression_model_settings(self):
        """...Test LinearRegression setting of parameters of model
        """
        for solver in LinearRegression._solvers.keys():
            learner = LinearRegression(fit_intercept=True, solver=solver)
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)
            learner.fit_intercept = False
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)

            learner = LinearRegression(fit_intercept=False, solver=solver)
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)
            learner.fit_intercept = True
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)

    def test_LinearRegression_penalty_C(self):
        """...Test LinearRegression setting of parameter of C
        """
        for penalty in LinearRegression._penalties.keys():
            if penalty != 'none':
                if penalty == 'binarsity':
                    learner = LinearRegression(penalty=penalty, C=self.float_1,
                                               blocks_start=[0],
                                               blocks_length=[1])
                else:
                    learner = LinearRegression(penalty=penalty, C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    if penalty == 'binarsity':
                        LinearRegression(penalty=penalty, C=-1,
                                         blocks_start=[0], blocks_length=[1])
                    else:
                        LinearRegression(penalty=penalty, C=-1)
            else:
                pass
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    LinearRegression(penalty=penalty, C=self.float_1)

                learner = LinearRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_LinearRegression_penalty_elastic_net_ratio(self):
        """...Test LinearRegression setting of parameter of elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3

        for penalty in LinearRegression._penalties.keys():
            if penalty == 'elasticnet':
                learner = LinearRegression(penalty=penalty, C=self.float_1,
                                           elastic_net_ratio=ratio_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.elastic_net_ratio, ratio_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                self.assertEqual(learner._prox_obj.ratio, ratio_1)

                learner.elastic_net_ratio = ratio_2
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.elastic_net_ratio, ratio_2)
                self.assertEqual(learner._prox_obj.ratio, ratio_2)
            else:
                msg = '^Penalty "%s" has no elastic_net_ratio attribute$$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    if penalty == 'binarsity':
                        LinearRegression(penalty=penalty,
                                         elastic_net_ratio=0.8,
                                         blocks_start=[0], blocks_length=[1])
                    else:
                        LinearRegression(penalty=penalty,
                                         elastic_net_ratio=0.8)

                if penalty == 'binarsity':
                    learner = LinearRegression(
                        penalty=penalty, blocks_start=[0], blocks_length=[1])
                else:
                    learner = LinearRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_LinearRegression_solver_basic_settings(self):
        """...Test LinearRegression setting of basic parameters of solver
        """
        for solver in LinearRegression._solvers.keys():
            # tol
            learner = LinearRegression(solver=solver, tol=self.float_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = LinearRegression(solver=solver, max_iter=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = LinearRegression(solver=solver, verbose=True,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = LinearRegression(solver=solver, verbose=False,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = LinearRegression(solver=solver, print_every=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = LinearRegression(solver=solver, record_every=self.int_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_LinearRegression_solver_step(self):
        """...Test LinearRegression setting of step parameter of solver
        """
        for solver in LinearRegression._solvers.keys():
            learner = LinearRegression(solver=solver, step=self.float_1,
                                       **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.step, self.float_1)
            self.assertEqual(learner._solver_obj.step, self.float_1)
            learner.step = self.float_2
            self.assertEqual(learner.step, self.float_2)
            self.assertEqual(learner._solver_obj.step, self.float_2)

    def test_LinearRegression_solver_random_state(self):
        """...Test LinearRegression setting of random_state parameter of solver
        """
        for solver in LinearRegression._solvers.keys():
            if solver in ['agd', 'gd']:
                msg = '^Solver "%s" has no settable random_state$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = LinearRegression(
                        solver=solver, random_state=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.random_state)
            else:
                learner = LinearRegression(
                    solver=solver, random_state=self.int_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)

                msg = '^random_state must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    LinearRegression(solver=solver, random_state=-1,
                                     **Test.specific_solver_kwargs(solver))

            msg = '^random_state is readonly in LinearRegression$'
            with self.assertRaisesRegex(AttributeError, msg):
                learner = LinearRegression(
                    solver=solver, **Test.specific_solver_kwargs(solver))
                learner.random_state = self.int_2

    def test_safe_array_cast(self):
        """...Test error and warnings raised by LogLearner constructor
        """
        msg = '^Copying array of size \(5, 5\) to convert it in the ' \
              'right format$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            LinearRegression._safe_array(self.X.astype(int))

        msg = '^Copying array of size \(3, 5\) to create a ' \
              'C-contiguous version of it$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            LinearRegression._safe_array(self.X[::2])

        np.testing.assert_array_equal(self.X,
                                      LinearRegression._safe_array(self.X))

    def test_predict(self):
        """...Test LinearRegression predict
        """
        X_train, y_train, _, _ = self.get_train_data(n_samples=200,
                                                     n_features=12)
        learner = LinearRegression(random_state=32789, tol=1e-9)
        learner.fit(X_train, y_train)
        X_test, y_test, _, _ = self.get_train_data(n_samples=5, n_features=12)
        y_pred = np.array([0.084, -1.4276, -3.1555, 2.6218, 0.3736])
        np.testing.assert_array_almost_equal(
            learner.predict(X_test), y_pred, decimal=4)

    def test_score(self):
        """...Test LinearRegression predict
        """
        X_train, y_train, _, _ = self.get_train_data(n_samples=2000,
                                                     n_features=12)
        learner = LinearRegression(random_state=32789, tol=1e-9)
        learner.fit(X_train, y_train)
        X_test, y_test, _, _ = self.get_train_data(n_samples=200,
                                                   n_features=12)
        self.assertAlmostEqual(
            learner.score(X_test, y_test), 0.793774, places=4)


if __name__ == "__main__":
    unittest.main()
