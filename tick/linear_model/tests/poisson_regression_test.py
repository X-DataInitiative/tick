# License: BSD 3 clause

import unittest
from itertools import product

import numpy as np

from tick.base.inference import InferenceTest
from tick.linear_model import SimuPoisReg, PoissonRegression
from tick.simulation import weights_sparse_gauss


class Test(InferenceTest):
    def setUp(self):
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230

        self.X = np.zeros((5, 5))

    @staticmethod
    def get_train_data(n_samples=2000, n_features=20, fit_intercept=True):
        np.random.seed(123)
        weights0 = weights_sparse_gauss(n_features, nnz=2)
        if fit_intercept:
            intercept0 = 1.
        else:
            intercept0 = None
        X, y = SimuPoisReg(weights0, intercept0, n_samples=n_samples, seed=123,
                           verbose=False).simulate()
        return X, y, weights0, intercept0

    def test_PoissonRegression_run(self):
        """...Test PoissonRegression runs with different solvers and penalties
        """
        n_samples = 200
        n_features = 10

        for fit_intercept in [False, True]:
            X, y, weights0, intercept0 = self.get_train_data(
                n_samples=n_samples, n_features=n_features,
                fit_intercept=fit_intercept)
            for solver, penalty in product(PoissonRegression._solvers,
                                           PoissonRegression._penalties):
                if solver == 'bfgs' and (penalty not in ['zero', 'l2']):
                    continue

                if penalty == 'binarsity':
                    learner = PoissonRegression(
                        verbose=False, fit_intercept=fit_intercept,
                        solver=solver, penalty=penalty, max_iter=1, step=1e-5,
                        blocks_start=[0], blocks_length=[1])
                else:
                    learner = PoissonRegression(
                        verbose=False, fit_intercept=fit_intercept,
                        solver=solver, penalty=penalty, max_iter=1, step=1e-5)

                learner.fit(X, y)
                self.assertTrue(np.isfinite(learner.weights).all())
                if fit_intercept:
                    self.assertTrue(np.isfinite(learner.intercept))

    def test_PoissonRegression_fit(self):
        """...Test PoissonRegression fit with default parameters
        """
        n_samples = 2000
        n_features = 20

        for fit_intercept in [False, True]:
            X, y, weights0, intercept0 = self.get_train_data(
                n_samples=n_samples, n_features=n_features,
                fit_intercept=fit_intercept)

            learner = PoissonRegression(C=1e3, verbose=False,
                                        fit_intercept=fit_intercept,
                                        solver='bfgs')
            learner.fit(X, y)
            err = np.linalg.norm(learner.weights - weights0) / n_features
            self.assertLess(err, 1e-2)
            if fit_intercept:
                self.assertLess(np.abs(learner.intercept - intercept0), 1e-1)

    def test_PoissonRegression_settings(self):
        """...Test PoissonRegression basic settings
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
        solver_class_map = PoissonRegression._solvers
        for solver in PoissonRegression._solvers.keys():
            learner = PoissonRegression(solver=solver)
            solver_class = solvers[solver_class_map[solver]]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

        msg = '^``solver`` must be one of agd, bfgs, gd, sgd, svrg, ' \
              'got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            PoissonRegression(solver='wrong_name')

        prox_class_map = PoissonRegression._penalties
        for penalty in PoissonRegression._penalties.keys():
            if penalty == 'binarsity':
                learner = PoissonRegression(penalty=penalty, blocks_start=[0],
                                            blocks_length=[1])
            else:
                learner = PoissonRegression(penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

        msg = '^``penalty`` must be one of binarsity, elasticnet, l1, l2, ' \
              'none, tv, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            PoissonRegression(penalty='wrong_name')

    def test_PoissonRegression_model_settings(self):
        """...Test LogisticRegression setting of parameters of model
        """
        for solver in PoissonRegression._solvers.keys():
            learner = PoissonRegression(fit_intercept=True, solver=solver)
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)
            learner.fit_intercept = False
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)

            learner = PoissonRegression(fit_intercept=False, solver=solver)
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)
            learner.fit_intercept = True
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)

    def test_PoissonRegression_penalty_C(self):
        """...Test PoissonRegression setting of parameter of C
        """
        for penalty in PoissonRegression._penalties.keys():
            if penalty != 'none':
                if penalty == 'binarsity':
                    learner = PoissonRegression(
                        penalty=penalty, C=self.float_1, blocks_start=[0],
                        blocks_length=[1])
                else:
                    learner = PoissonRegression(penalty=penalty,
                                                C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    if penalty == 'binarsity':
                        PoissonRegression(penalty=penalty, C=-1,
                                          blocks_start=[0], blocks_length=[1])
                    else:
                        PoissonRegression(penalty=penalty, C=-1)
            else:
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    PoissonRegression(penalty=penalty, C=self.float_1)

                learner = PoissonRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_PoissonRegression_penalty_elastic_net_ratio(self):
        """...Test PoissonRegression setting of parameter of elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3
        for penalty in PoissonRegression._penalties.keys():
            if penalty == 'elasticnet':
                learner = PoissonRegression(penalty=penalty, C=self.float_1,
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
                        PoissonRegression(penalty=penalty,
                                          elastic_net_ratio=0.8,
                                          blocks_start=[0], blocks_length=[1])
                    else:
                        PoissonRegression(penalty=penalty,
                                          elastic_net_ratio=0.8)

                if penalty == 'binarsity':
                    learner = PoissonRegression(
                        penalty=penalty, blocks_start=[0], blocks_length=[1])
                else:
                    learner = PoissonRegression(penalty=penalty)

                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_PoissonRegression_solver_basic_settings(self):
        """...Test LogisticRegression setting of basic parameters of solver
        """
        for solver in PoissonRegression._solvers.keys():
            # tol
            learner = PoissonRegression(solver=solver, tol=self.float_1)
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = PoissonRegression(solver=solver, max_iter=self.int_1)
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = PoissonRegression(solver=solver, verbose=True)
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = PoissonRegression(solver=solver, verbose=False)
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = PoissonRegression(solver=solver, print_every=self.int_1)
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = PoissonRegression(solver=solver, record_every=self.int_1)
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_PoissonRegression_solver_step(self):
        """...Test LogisticRegression setting of step parameter of solver
        """
        for solver in PoissonRegression._solvers.keys():
            if solver == 'bfgs':
                learner = PoissonRegression(solver=solver)
                self.assertIsNone(learner.step)
                learner = PoissonRegression(solver=solver, step=self.float_1)
                self.assertIsNone(learner.step)
                msg = '^Solver "bfgs" has no settable step$'
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.step = self.float_2
                    self.assertIsNone(learner.step)
            else:
                learner = PoissonRegression(solver=solver, step=self.float_1)
                self.assertEqual(learner.step, self.float_1)
                self.assertEqual(learner._solver_obj.step, self.float_1)
                learner.step = self.float_2
                self.assertEqual(learner.step, self.float_2)
                self.assertEqual(learner._solver_obj.step, self.float_2)

    def test_PoissonRegression_solver_random_state(self):
        """...Test PoissonRegression setting of random_state parameter of solver
        """
        for solver in PoissonRegression._solvers.keys():
            if solver in ['agd', 'gd', 'bfgs']:
                msg = '^Solver "%s" has no settable random_state$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = PoissonRegression(solver=solver, random_state=1)

                    self.assertIsNone(learner.random_state)
            else:
                learner = PoissonRegression(solver=solver,
                                            random_state=self.int_1)
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)

                msg = '^random_state must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    PoissonRegression(solver=solver, random_state=-1)

            msg = '^random_state is readonly in PoissonRegression$'
            with self.assertRaisesRegex(AttributeError, msg):
                learner = PoissonRegression(solver=solver)
                learner.random_state = self.int_2

    def test_safe_array_cast(self):
        """...Test error and warnings raised by LogLearner constructor
        """
        msg = '^Copying array of size \(5, 5\) to convert it in the ' \
              'right format$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            PoissonRegression._safe_array(self.X.astype(int))

        msg = '^Copying array of size \(3, 5\) to create a ' \
              'C-contiguous version of it$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            PoissonRegression._safe_array(self.X[::2])

        np.testing.assert_array_equal(self.X,
                                      PoissonRegression._safe_array(self.X))

    def test_predict(self):
        """...Test PoissonRegression predict
        """
        X_train, y_train, _, _ = self.get_train_data(n_samples=200,
                                                     n_features=12)
        learner = PoissonRegression(random_state=32789, tol=1e-9)
        learner.fit(X_train, y_train)
        X_test, y_test, _, _ = self.get_train_data(n_samples=5, n_features=12)
        y_pred = np.array([1., 5., 0., 5., 6.])
        np.testing.assert_array_almost_equal(learner.predict(X_test), y_pred)

    def test_decision_function(self):
        """...Test PoissonRegression decision function
        """
        X_train, y_train, _, _ = self.get_train_data(n_samples=200,
                                                     n_features=12)
        learner = PoissonRegression(random_state=32789, tol=1e-9)
        learner.fit(X_train, y_train)
        X_test, y_test, _, _ = self.get_train_data(n_samples=5, n_features=12)
        y_pred = np.array([1.1448, 5.2194, 0.2624, 4.5525, 6.4168])
        np.testing.assert_array_almost_equal(
            learner.decision_function(X_test), y_pred, decimal=4)

    def test_loglik(self):
        """...Test PoissonRegression loglik function
        """
        X_train, y_train, _, _ = self.get_train_data(n_samples=200,
                                                     n_features=12)
        learner = PoissonRegression(random_state=32789, tol=1e-9)
        learner.fit(X_train, y_train)
        X_test, y_test, _, _ = self.get_train_data(n_samples=5, n_features=12)
        np.testing.assert_array_almost_equal(
            learner.loglik(X_test, y_test), 1.8254, decimal=4)


if __name__ == "__main__":
    unittest.main()
