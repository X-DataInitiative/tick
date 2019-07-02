# License: BSD 3 clause

import itertools
import unittest

import numpy as np

from tick.base.inference import InferenceTest
from tick.robust import RobustLinearRegression
from tick.simulation import weights_sparse_gauss, features_normal_cov_toeplitz
from tick.metrics import support_fdp, support_recall


class Test(InferenceTest):
    n_samples = 300
    n_features = 5
    noise_level = 1.
    nnz_outliers = 10
    outliers_intensity = 5.
    interc0 = -3.
    target_fdr = 0.2

    def setUp(self):
        self.float_1 = 5.23e-4
        self.float_2 = 3.86e-2
        self.int_1 = 3198
        self.int_2 = 230
        self.X = np.zeros((5, 5))

    @staticmethod
    def get_train_data(fit_intercept=True):
        np.random.seed(12)
        n_samples = Test.n_samples
        n_features = Test.n_features
        noise_level = Test.noise_level
        nnz_outliers = Test.nnz_outliers
        outliers_intensity = Test.outliers_intensity
        if fit_intercept:
            interc0 = Test.interc0
        else:
            interc0 = 0.
        weights0 = np.sqrt(
            2 * np.log(np.linspace(1, 10, n_features) * n_features))
        sample_intercepts0 = weights_sparse_gauss(n_weights=n_samples,
                                                  nnz=nnz_outliers)
        sample_intercepts0[sample_intercepts0 != 0] \
            = outliers_intensity \
              * np.sqrt(2 * np.log(np.linspace(1, 10, nnz_outliers)
                                   * n_samples)) \
              * np.sign(sample_intercepts0[sample_intercepts0 != 0])

        X = features_normal_cov_toeplitz(n_samples, n_features, 0.5)
        y = X.dot(weights0) + noise_level * np.random.randn(n_samples) + interc0
        y += sample_intercepts0
        return X, y, weights0, interc0, sample_intercepts0

    def test_RobustLinearRegression_fit(self):
        """...Test RobustLinearRegression fit with different solvers and penalties
        """
        X, y, weights0, interc0, sample_intercepts0 = self.get_train_data(True)

        solvers = RobustLinearRegression._solvers.keys()
        for i, solver in enumerate(solvers):
            learner_keywords = {
                'C_sample_intercepts': Test.n_samples / Test.noise_level,
                'fit_intercept': True,
                'fdr': Test.target_fdr,
                'max_iter': 3000,
                'tol': 1e-7,
                'solver': solver,
                'penalty': 'none',
                'verbose': False
            }

            learner = RobustLinearRegression(**learner_keywords)
            learner.fit(X, y)

            weights = [
                1.82145051, 2.32011366, 2.6886905, 2.53289584, 2.86991904
            ]
            interc = -2.9877245464563931
            fdp_ = 0.23076923076923078
            power = 1.0

            np.testing.assert_array_almost_equal(weights, learner.weights, 2)
            self.assertAlmostEqual(interc, learner.intercept, 2)
            self.assertAlmostEqual(
                fdp_, support_fdp(sample_intercepts0,
                                  learner.sample_intercepts), 4)
            self.assertAlmostEqual(
                power,
                support_recall(sample_intercepts0, learner.sample_intercepts),
                4)

        X, y, weights0, interc0, sample_intercepts0 = self.get_train_data(
            False)
        for i, solver in enumerate(solvers):
            learner_keywords = {
                'C_sample_intercepts': Test.n_samples / Test.noise_level,
                'fit_intercept': False,
                'fdr': Test.target_fdr,
                'max_iter': 3000,
                'tol': 1e-7,
                'solver': solver,
                'verbose': False
            }
            learner = RobustLinearRegression(**learner_keywords)
            learner.fit(X, y)
            weights = [
                1.82341444, 2.3226882, 2.68081823, 2.53942366, 2.86439685
            ]
            interc = None
            fdp_ = 0.23076923076923078
            power = 1.0

            np.testing.assert_array_almost_equal(weights, learner.weights, 2)
            self.assertEqual(interc, learner.intercept)
            self.assertAlmostEqual(
                fdp_, support_fdp(sample_intercepts0,
                                  learner.sample_intercepts), 4)
            self.assertAlmostEqual(
                power,
                support_recall(sample_intercepts0, learner.sample_intercepts),
                4)

    def test_RobustLinearRegression_warm_start(self):
        """...Test RobustLinearRegression warm start
        """
        X, y, weights0, interc0, sample_intercepts0 = self.get_train_data(True)
        fit_intercepts = [True, False]
        cases = itertools.product(RobustLinearRegression._solvers.keys(),
                                  fit_intercepts)
        for solver, fit_intercept in cases:
            solver_kwargs = {
                'C_sample_intercepts': Test.n_samples / Test.noise_level,
                'solver': solver,
                'max_iter': 2,
                'verbose': False,
                'fit_intercept': fit_intercept,
                'warm_start': True,
                'tol': 0
            }
            learner = RobustLinearRegression(**solver_kwargs)
            learner.fit(X, y)
            coeffs_1 = learner.coeffs
            learner.fit(X, y)
            coeffs_2 = learner.coeffs
            # Thanks to warm start objective should have decreased
            self.assertLess(
                learner._solver_obj.objective(coeffs_2),
                learner._solver_obj.objective(coeffs_1))

    @staticmethod
    def specific_solver_kwargs(solver):
        """...A simple method to as systematically some kwargs to our tests
        """
        return dict()

    def test_RobustLinearRegression_settings(self):
        """...Test RobustLinearRegression basic settings
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
        solver_class_map = RobustLinearRegression._solvers
        for solver in RobustLinearRegression._solvers.keys():
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver,
                **Test.specific_solver_kwargs(solver))
            solver_class = solvers[solver_class_map[solver]]
            self.assertTrue(isinstance(learner._solver_obj, solver_class))

        msg = '^``solver`` must be one of agd, gd, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., solver='wrong_name')
        # prox
        prox_class_map = RobustLinearRegression._penalties
        for penalty in RobustLinearRegression._penalties.keys():
            learner = RobustLinearRegression(C_sample_intercepts=1.,
                                             penalty=penalty)
            prox_class = prox_class_map[penalty]
            self.assertTrue(isinstance(learner._prox_obj, prox_class))

        msg = '^``penalty`` must be one of elasticnet, l1, l2, ' \
              'none, slope, got wrong_name$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1.,
                                   penalty='wrong_name')

    def test_RobustLinearRegression_model_settings(self):
        """...Test RobustLinearRegression setting of parameters of model
        """
        for solver in RobustLinearRegression._solvers.keys():
            learner = RobustLinearRegression(C_sample_intercepts=1.,
                                             fit_intercept=True, solver=solver)
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)
            learner.fit_intercept = False
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)

            learner = RobustLinearRegression(
                C_sample_intercepts=1., fit_intercept=False, solver=solver)
            self.assertEqual(learner.fit_intercept, False)
            self.assertEqual(learner._model_obj.fit_intercept, False)
            learner.fit_intercept = True
            self.assertEqual(learner.fit_intercept, True)
            self.assertEqual(learner._model_obj.fit_intercept, True)

    def test_RobustLinearRegression_penalty_C(self):
        """...Test RobustLinearRegression setting of parameter of C
        """
        for penalty in RobustLinearRegression._penalties.keys():
            if penalty != 'none':
                learner = RobustLinearRegression(
                    C_sample_intercepts=1., penalty=penalty, C=self.float_1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                learner.C = self.float_2
                self.assertEqual(learner.C, self.float_2)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_2)

                msg = '^``C`` must be positive, got -1$'
                with self.assertRaisesRegex(ValueError, msg):
                    RobustLinearRegression(C_sample_intercepts=1.,
                                           penalty=penalty, C=-1)
            else:
                pass
                msg = '^You cannot set C for penalty "%s"$' % penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    RobustLinearRegression(C_sample_intercepts=1.,
                                           penalty=penalty, C=self.float_1)

                learner = RobustLinearRegression(C_sample_intercepts=1.,
                                                 penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.C = self.float_1

            msg = '^``C`` must be positive, got -2$'
            with self.assertRaisesRegex(ValueError, msg):
                learner.C = -2

    def test_RobustLinearRegression_penalty_C_sample_intercepts(self):
        """...Test RobustLinearRegression setting of parameter of C
        """
        learner = RobustLinearRegression(C_sample_intercepts=self.float_1)
        self.assertEqual(learner.C_sample_intercepts, self.float_1)
        self.assertEqual(learner._prox_intercepts_obj.strength,
                         1. / self.float_1)
        learner.C_sample_intercepts = self.float_2
        self.assertEqual(learner.C_sample_intercepts, self.float_2)
        self.assertEqual(learner._prox_intercepts_obj.strength,
                         1. / self.float_2)

        msg = '^``C_sample_intercepts`` must be positive, got -1.0$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=-1.)

        msg = '^``C_sample_intercepts`` cannot be `None`$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=None)

        msg = '^``C_sample_intercepts`` cannot be 0.$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=0.)

        msg = '^``C_sample_intercepts`` must be a finite number, got inf$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=np.inf)

    def test_RobustLinearRegression_fdr(self):
        """...Test RobustLinearRegression setting of parameter of C
        """
        fdr1 = 0.33
        fdr2 = 0.78
        learner = RobustLinearRegression(C_sample_intercepts=self.float_1,
                                         fdr=fdr1)
        self.assertEqual(learner.fdr, fdr1)
        self.assertEqual(learner._prox_intercepts_obj.fdr, fdr1)
        learner.fdr = fdr2
        self.assertEqual(learner.fdr, fdr2)
        self.assertEqual(learner._prox_intercepts_obj.fdr, fdr2)

        msg = '^``fdr`` must be in \(0, 1\), got -1.0'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., fdr=-1.0)

        msg = '^``fdr`` must be in \(0, 1\), got 1.5'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., fdr=1.5)

        msg = '^``fdr`` cannot be `None`$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., fdr=None)

        msg = '^``fdr`` must be a finite number, got inf$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., fdr=np.inf)

    def test_RobustLinearRegression_penalty_elastic_net_ratio(self):
        """...Test RobustLinearRegression setting of parameter of elastic_net_ratio
        """
        ratio_1 = 0.6
        ratio_2 = 0.3
        for penalty in RobustLinearRegression._penalties.keys():
            if penalty == 'elasticnet':
                learner = RobustLinearRegression(
                    C_sample_intercepts=1., penalty=penalty, C=self.float_1,
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
                    RobustLinearRegression(C_sample_intercepts=1.,
                                           penalty=penalty,
                                           elastic_net_ratio=0.8)

                learner = RobustLinearRegression(C_sample_intercepts=1.,
                                                 penalty=penalty)
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.elastic_net_ratio = ratio_1

    def test_RobustLinearRegression_penalty_slope_fdr(self):
        """...Test RobustLinearRegression setting of parameter of slope_fdr
        """
        slope_fdr1 = 0.6
        slope_fdr2 = 0.3
        for penalty in RobustLinearRegression._penalties.keys():
            if penalty == 'slope':
                learner = RobustLinearRegression(
                    C_sample_intercepts=1., penalty=penalty, C=self.float_1,
                    slope_fdr=slope_fdr1)
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.slope_fdr, slope_fdr1)
                self.assertEqual(learner._prox_obj.strength, 1. / self.float_1)
                self.assertEqual(learner._prox_obj.fdr, slope_fdr1)

                learner.slope_fdr = slope_fdr2
                self.assertEqual(learner.C, self.float_1)
                self.assertEqual(learner.slope_fdr, slope_fdr2)
                self.assertEqual(learner._prox_obj.fdr, slope_fdr2)
            else:
                msg = '^Penalty "%s" has no ``slope_fdr`` attribute$$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    RobustLinearRegression(C_sample_intercepts=1.,
                                           penalty=penalty, slope_fdr=0.8)

                learner = RobustLinearRegression(C_sample_intercepts=1.,
                                                 penalty=penalty)
                msg = '^Penalty "%s" has no ``slope_fdr`` attribute$$' % \
                      penalty
                with self.assertWarnsRegex(RuntimeWarning, msg):
                    learner.slope_fdr = slope_fdr1

        slope_fdr1 = 0.33
        slope_fdr2 = 0.78
        learner = RobustLinearRegression(C_sample_intercepts=self.float_1,
                                         slope_fdr=slope_fdr1, penalty='slope')
        self.assertEqual(learner.slope_fdr, slope_fdr1)
        self.assertEqual(learner._prox_obj.fdr, slope_fdr1)
        learner.slope_fdr = slope_fdr2
        self.assertEqual(learner.slope_fdr, slope_fdr2)
        self.assertEqual(learner._prox_obj.fdr, slope_fdr2)

        msg = '^``slope_fdr`` must be in \(0, 1\), got -1.0'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., penalty='slope',
                                   slope_fdr=-1.0)

        msg = '^``slope_fdr`` must be in \(0, 1\), got 1.5'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., penalty='slope',
                                   slope_fdr=1.5)

        msg = '^``slope_fdr`` cannot be `None`$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., penalty='slope',
                                   slope_fdr=None)

        msg = '^``slope_fdr`` must be a finite number, got inf$'
        with self.assertRaisesRegex(ValueError, msg):
            RobustLinearRegression(C_sample_intercepts=1., penalty='slope',
                                   slope_fdr=np.inf)

    def test_RobustLinearRegression_solver_basic_settings(self):
        """...Test RobustLinearRegression setting of basic parameters of solver
        """
        for solver in RobustLinearRegression._solvers.keys():
            # tol
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, tol=self.float_1,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.tol, self.float_1)
            self.assertEqual(learner._solver_obj.tol, self.float_1)
            learner.tol = self.float_2
            self.assertEqual(learner.tol, self.float_2)
            self.assertEqual(learner._solver_obj.tol, self.float_2)

            # max_iter
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, max_iter=self.int_1,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.max_iter, self.int_1)
            self.assertEqual(learner._solver_obj.max_iter, self.int_1)
            learner.max_iter = self.int_2
            self.assertEqual(learner.max_iter, self.int_2)
            self.assertEqual(learner._solver_obj.max_iter, self.int_2)

            # verbose
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, verbose=True,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)
            learner.verbose = False
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)

            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, verbose=False,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.verbose, False)
            self.assertEqual(learner._solver_obj.verbose, False)
            learner.verbose = True
            self.assertEqual(learner.verbose, True)
            self.assertEqual(learner._solver_obj.verbose, True)

            # print_every
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, print_every=self.int_1,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.print_every, self.int_1)
            self.assertEqual(learner._solver_obj.print_every, self.int_1)
            learner.print_every = self.int_2
            self.assertEqual(learner.print_every, self.int_2)
            self.assertEqual(learner._solver_obj.print_every, self.int_2)

            # record_every
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, record_every=self.int_1,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.record_every, self.int_1)
            self.assertEqual(learner._solver_obj.record_every, self.int_1)
            learner.record_every = self.int_2
            self.assertEqual(learner.record_every, self.int_2)
            self.assertEqual(learner._solver_obj.record_every, self.int_2)

    def test_RobustLinearRegression_solver_step(self):
        """...Test RobustLinearRegression setting of step parameter of solver
        """
        for solver in RobustLinearRegression._solvers.keys():
            learner = RobustLinearRegression(
                C_sample_intercepts=1., solver=solver, step=self.float_1,
                **Test.specific_solver_kwargs(solver))
            self.assertEqual(learner.step, self.float_1)
            self.assertEqual(learner._solver_obj.step, self.float_1)
            learner.step = self.float_2
            self.assertEqual(learner.step, self.float_2)
            self.assertEqual(learner._solver_obj.step, self.float_2)

    def test_safe_array_cast(self):
        """...Test error and warnings raised by LogLearner constructor
        """
        msg = '^Copying array of size \(5, 5\) to convert it in the ' \
              'right format$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            RobustLinearRegression._safe_array(self.X.astype(int))

        msg = '^Copying array of size \(3, 5\) to create a ' \
              'C-contiguous version of it$'
        with self.assertWarnsRegex(RuntimeWarning, msg):
            RobustLinearRegression._safe_array(self.X[::2])

        np.testing.assert_array_equal(
            self.X, RobustLinearRegression._safe_array(self.X))


if __name__ == "__main__":
    unittest.main()
