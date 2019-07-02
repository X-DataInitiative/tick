# License: BSD 3 clause

import itertools
import unittest

import numpy as np
from sklearn.metrics.ranking import roc_auc_score

from tick.base.inference import InferenceTest
from tick.linear_model import SimuLogReg, LogisticRegression
from tick.simulation import weights_sparse_gauss
from tick.preprocessing.features_binarizer import FeaturesBinarizer
from tick.prox import ProxZero, ProxL1, ProxL2Sq, ProxElasticNet, ProxTV, \
    ProxBinarsity

solvers = ['gd', 'agd', 'sgd', 'sdca', 'bfgs', 'svrg']
penalties = ['none', 'l2', 'l1', 'tv', 'elasticnet', 'binarsity']


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
    def get_train_data(n_features=20, n_samples=3000, nnz=5):
        np.random.seed(12)
        weights0 = weights_sparse_gauss(n_features, nnz=nnz)
        interc0 = 0.1
        features, y = SimuLogReg(weights0, interc0, n_samples=n_samples,
                                 verbose=False).simulate()
        return features, y

    def test_LogisticRegression_fit(self):
        """...Test LogisticRegression fit with different solvers and penalties
        """
        sto_seed = 179312
        raw_features, y = Test.get_train_data()

        for fit_intercept in [True, False]:
            for penalty in penalties:

                if penalty == 'binarsity':
                    # binarize features
                    n_cuts = 3
                    binarizer = FeaturesBinarizer(n_cuts=n_cuts)
                    features = binarizer.fit_transform(raw_features)
                else:
                    features = raw_features

                for solver in solvers:
                    solver_kwargs = {
                        'penalty': penalty,
                        'tol': 1e-5,
                        'solver': solver,
                        'verbose': False,
                        'max_iter': 10,
                        'fit_intercept': fit_intercept
                    }

                    if penalty != 'none':
                        solver_kwargs['C'] = 100

                    if penalty == 'binarsity':
                        solver_kwargs['blocks_start'] = binarizer.blocks_start
                        solver_kwargs[
                            'blocks_length'] = binarizer.blocks_length

                    if solver == 'sdca':
                        solver_kwargs['sdca_ridge_strength'] = 2e-2

                    if solver in ['sgd', 'svrg', 'sdca']:
                        solver_kwargs['random_state'] = sto_seed

                    if solver == 'sgd':
                        solver_kwargs['step'] = 1.

                    if solver == 'bfgs':
                        # BFGS only accepts ProxZero and ProxL2sq for now
                        if penalty not in ['none', 'l2']:
                            continue

                    learner = LogisticRegression(**solver_kwargs)
                    learner.fit(features, y)
                    probas = learner.predict_proba(features)[:, 1]
                    auc = roc_auc_score(y, probas)
                    self.assertGreater(
                        auc, 0.7, "solver %s with penalty %s and "
                        "intercept %s reached too low AUC" % (solver, penalty,
                                                              fit_intercept))

    def test_LogisticRegression_warm_start(self):
        """...Test LogisticRegression warm start
        """
        sto_seed = 179312
        X, y = Test.get_train_data()

        fit_intercepts = [True, False]
        cases = itertools.product(solvers, fit_intercepts)

        for solver, fit_intercept in cases:
            solver_kwargs = {
                'solver': solver,
                'max_iter': 2,
                'fit_intercept': fit_intercept,
                'warm_start': True,
                'tol': 0
            }

            if solver == 'sdca':
                msg = '^SDCA cannot be warm started$'
                with self.assertRaisesRegex(ValueError, msg):
                    LogisticRegression(**solver_kwargs)

            else:

                if solver in ['sgd', 'svrg']:
                    solver_kwargs['random_state'] = sto_seed

                if solver == 'sgd':
                    solver_kwargs['step'] = .3

                learner = LogisticRegression(**solver_kwargs)

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

    def test_LogisticRegression_settings(self):
        """...Test LogisticRegression basic settings
        """
        # solver
        from tick.solver import AGD, GD, BFGS, SGD, SVRG, SDCA
        solver_class_map = {
            'gd': GD,
            'agd': AGD,
            'sgd': SGD,
            'svrg': SVRG,
            'bfgs': BFGS,
            'sdca': SDCA
        }
        for solver in solvers:
            learner = LogisticRegression(solver=solver,
                                         **Test.specific_solver_kwargs(solver))
            solver_class = solver_class_map[solver]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

        msg = '^``solver`` must be one of agd, bfgs, gd, sdca, sgd, ' \
              'svrg, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            LogisticRegression(solver='wrong_name')

        # prox
        prox_class_map = {
            'none': ProxZero,
            'l1': ProxL1,
            'l2': ProxL2Sq,
            'elasticnet': ProxElasticNet,
            'tv': ProxTV,
            'binarsity': ProxBinarsity
        }
        for penalty in penalties:
            if penalty == 'binarsity':
                learner = LogisticRegression(penalty=penalty, blocks_start=[0],
                                             blocks_length=[1])
            else:
                learner = LogisticRegression(penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

        msg = '^``penalty`` must be one of binarsity, elasticnet, l1, l2, none, ' \
              'tv, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            LogisticRegression(penalty='wrong_name')

    def test_LogisticRegression_model_settings(self):
        """...Test LogisticRegression setting of parameters of model
        """
        for solver in solvers:
            learner = LogisticRegression(fit_intercept=True, solver=solver)
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)
            learner.fit_intercept = False
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)

            learner = LogisticRegression(fit_intercept=False, solver=solver)
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)
            learner.fit_intercept = True
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)

    def test_LogisticRegression_penalty_C(self):
        """...Test LogisticRegression setting of parameter of C
        """
        for penalty in penalties:
            if penalty != 'none':
                if penalty == 'binarsity':
                    learner = LogisticRegression(
                        penalty=penalty, C=self.float_1, blocks_start=[0],
                        blocks_length=[1])
                else:
                    learner = LogisticRegression(penalty=penalty,
                                                 C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)
                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    if penalty == 'binarsity':
                        LogisticRegression(penalty=penalty, C=-1,
                                           blocks_start=[0], blocks_length=[1])
                    else:
                        LogisticRegression(penalty=penalty, C=-1)
            else:
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    if penalty == 'binarsity':
                        LogisticRegression(penalty=penalty, C=self.float_1,
                                           blocks_start=[0], blocks_length=[1])
                    else:
                        LogisticRegression(penalty=penalty, C=self.float_1)

                if penalty == 'binarsity':
                    learner = LogisticRegression(
                        penalty=penalty, blocks_start=[0], blocks_length=[1])
                else:
                    learner = LogisticRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_LogisticRegression_penalty_elastic_net_ratio(self):
        """...Test LogisticRegression setting of parameter of elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3

        for penalty in penalties:
            if penalty == 'elasticnet':

                learner = LogisticRegression(penalty=penalty, C=self.float_1,
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
                        LogisticRegression(penalty=penalty,
                                           elastic_net_ratio=0.8,
                                           blocks_start=[0], blocks_length=[1])
                    else:
                        LogisticRegression(penalty=penalty,
                                           elastic_net_ratio=0.8)

                if penalty == 'binarsity':
                    learner = LogisticRegression(
                        penalty=penalty, blocks_start=[0], blocks_length=[1])
                else:
                    learner = LogisticRegression(penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_LogisticRegression_solver_basic_settings(self):
        """...Test LogisticRegression setting of basic parameters of solver
        """
        for solver in solvers:
            # tol
            learner = LogisticRegression(solver=solver, tol=self.float_1,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = LogisticRegression(solver=solver, max_iter=self.int_1,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = LogisticRegression(solver=solver, verbose=True,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = LogisticRegression(solver=solver, verbose=False,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = LogisticRegression(solver=solver, print_every=self.int_1,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = LogisticRegression(solver=solver,
                                         record_every=self.int_1,
                                         **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_LogisticRegression_solver_step(self):
        """...Test LogisticRegression setting of step parameter of solver
        """
        for solver in solvers:
            if solver in ['sdca', 'bfgs']:
                msg = '^Solver "%s" has no settable step$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = LogisticRegression(
                        solver=solver, step=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.step)
            else:
                learner = LogisticRegression(
                    solver=solver, step=self.float_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.step, self.float_1)
                self.assertEqual(learner._solver_obj.step, self.float_1)
                learner.step = self.float_2
                self.assertEqual(learner.step, self.float_2)
                self.assertEqual(learner._solver_obj.step, self.float_2)

            if solver in ['sgd']:
                msg = '^SGD step needs to be tuned manually$'
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = LogisticRegression(solver='sgd')
                    learner.fit(self.X, self.y)

    def test_LogisticRegression_solver_random_state(self):
        """...Test LogisticRegression setting of random_state parameter of solver
        """
        for solver in solvers:
            if solver in ['bfgs', 'agd', 'gd']:
                msg = '^Solver "%s" has no settable random_state$' % solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner = LogisticRegression(
                        solver=solver, random_state=1,
                        **Test.specific_solver_kwargs(solver))
                    self.assertIsNone(learner.random_state)
            else:
                learner = LogisticRegression(
                    solver=solver, random_state=self.int_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.random_state, self.int_1)
                self.assertEqual(learner._solver_obj.seed, self.int_1)

                msg = '^random_state must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    LogisticRegression(solver=solver, random_state=-1,
                                       **Test.specific_solver_kwargs(solver))

            msg = '^random_state is readonly in LogisticRegression$'
            with self.assertRaisesRegex(AttributeError, msg):
                learner = LogisticRegression(
                    solver=solver, **Test.specific_solver_kwargs(solver))
                learner.random_state = self.int_2

    def test_LogisticRegression_solver_sdca_ridge_strength(self):
        """...Test LogisticRegression setting of sdca_ridge_strength parameter
        of solver
        """
        for solver in solvers:
            if solver == 'sdca':
                learner = LogisticRegression(
                    solver=solver, sdca_ridge_strength=self.float_1,
                    **Test.specific_solver_kwargs(solver))
                self.assertEqual(learner.sdca_ridge_strength, self.float_1)
                self.assertEqual(learner._solver_obj._solver.get_l_l2sq(),
                                 self.float_1)

                learner.sdca_ridge_strength = self.float_2
                self.assertEqual(learner.sdca_ridge_strength, self.float_2)
                self.assertEqual(learner._solver_obj._solver.get_l_l2sq(),
                                 self.float_2)
            else:

                msg = '^Solver "%s" has no sdca_ridge_strength attribute$' % \
                      solver
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    LogisticRegression(solver=solver, sdca_ridge_strength=1e-2,
                                       **Test.specific_solver_kwargs(solver))

                learner = LogisticRegression(
                    solver=solver, **Test.specific_solver_kwargs(solver))
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.sdca_ridge_strength = self.float_1

    def test_safe_array_cast(self):
        """...Test error and warnings raised by LogLearner constructor
        """
        msg = '^Copying array of size \(5, 5\) to convert it in the ' \
              'right format$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            LogisticRegression._safe_array(self.X.astype(int))

        msg = '^Copying array of size \(3, 5\) to create a ' \
              'C-contiguous version of it$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            LogisticRegression._safe_array(self.X[::2])

        np.testing.assert_array_equal(self.X,
                                      LogisticRegression._safe_array(self.X))

    def test_labels_encoding(self):
        """...Test that class encoding is well done for LogReg
        """
        learner = LogisticRegression(max_iter=1)

        np.random.seed(38027)
        n_features = 3
        n_samples = 5
        X = np.random.rand(n_samples, n_features)

        encoded_y = np.array([1., -1., 1., -1., -1.])
        learner.fit(X, encoded_y)
        np.testing.assert_array_equal(learner.classes, [-1., 1.])
        np.testing.assert_array_equal(
            learner._encode_labels_vector(encoded_y), encoded_y)

        zero_one_y = np.array([1., 0., 1., 0., 0.])
        learner.fit(X, zero_one_y)
        np.testing.assert_array_equal(learner.classes, [0., 1.])
        np.testing.assert_array_equal(
            learner._encode_labels_vector(zero_one_y), encoded_y)

        text_y = np.array(['cat', 'dog', 'cat', 'dog', 'dog'])
        learner.fit(X, text_y)
        np.testing.assert_array_equal(set(learner.classes), {'cat', 'dog'})
        encoded_text_y = learner._encode_labels_vector(text_y)
        np.testing.assert_array_equal(
            encoded_text_y,
            encoded_y * np.sign(encoded_text_y[0]) * np.sign(encoded_y[0]))

    def test_predict(self):
        """...Test LogReg prediction
        """
        labels_mappings = [{
            -1: -1.,
            1: 1.
        }, {
            -1: 1.,
            1: -1.
        }, {
            -1: 1,
            1: 0
        }, {
            -1: 0,
            1: 1
        }, {
            -1: 'cat',
            1: 'dog'
        }]

        for labels_mapping in labels_mappings:
            X, y = Test.get_train_data(n_features=12, n_samples=300, nnz=0)
            y = np.vectorize(labels_mapping.get)(y)

            learner = LogisticRegression(random_state=32789, tol=1e-9)
            learner.fit(X, y)

            X_test, y_test = Test.get_train_data(n_features=12, n_samples=5,
                                                 nnz=0)
            predicted_y = [1., 1., -1., 1., 1.]
            predicted_y = np.vectorize(labels_mapping.get)(predicted_y)
            np.testing.assert_array_equal(learner.predict(X_test), predicted_y)

    def test_predict_proba(self):
        """...Test LogReg predict_proba
        """
        X, y = Test.get_train_data(n_features=12, n_samples=300, nnz=0)
        learner = LogisticRegression(random_state=32289, tol=1e-13)
        learner.fit(X, y)

        X_test, y_test = Test.get_train_data(n_features=12, n_samples=5, nnz=0)
        predicted_probas = np.array(
            [[0.35851418, 0.64148582], [0.42549328, 0.57450672],
             [0.6749705, 0.3250295], [0.39684181,
                                      0.60315819], [0.42732443, 0.57267557]])
        np.testing.assert_array_almost_equal(
            learner.predict_proba(X_test), predicted_probas, decimal=3)

    def test_decision_function(self):
        """...Test LogReg predict_proba
        """
        X, y = Test.get_train_data(n_features=12, n_samples=300, nnz=0)
        learner = LogisticRegression(random_state=32789, tol=1e-13)
        learner.fit(X, y)

        X_test, y_test = Test.get_train_data(n_features=12, n_samples=5, nnz=0)
        decision_function_values = np.array(
            [0.58182, 0.30026, -0.73075, 0.41864, 0.29278])
        np.testing.assert_array_almost_equal(
            learner.decision_function(X_test), decision_function_values,
            decimal=3)

    def test_float_double_arrays_fitting(self):
        X, y = Test.get_train_data(n_features=12, n_samples=300, nnz=0)
        learner_64 = LogisticRegression(random_state=32789, tol=1e-13)
        learner_64.fit(X, y)
        weights_64 = learner_64.weights
        self.assertEqual(weights_64.dtype, np.dtype('float64'))

        learner_32 = LogisticRegression(random_state=32789, tol=1e-13)
        X_32, y_32 = X.astype('float32'), y.astype('float32')
        learner_32.fit(X_32, y_32)
        weights_32 = learner_32.weights
        self.assertEqual(weights_32.dtype, np.dtype('float32'))

        np.testing.assert_array_almost_equal(weights_32, weights_64, decimal=5)


if __name__ == "__main__":
    unittest.main()
