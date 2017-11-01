# License: BSD 3 clause

import unittest
from tick.simulation import SimuLinReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np

from tick.inference import OnlineForestRegressor
from tick.inference.build.inference import Criterion_mse as mse
from tick.inference.build.inference import Criterion_unif as unif


class Test(unittest.TestCase):
    def test_online_forest_regression(self):
        n_samples = 500
        n_features = 3
        w0 = weights_sparse_gauss(n_features, nnz=2)
        X, y = SimuLinReg(w0, -1., n_samples=n_samples).simulate()
        X_train, X_test, y_train, y_test = train_test_split(X, y)

    def test_online_forest_regressor_n_threads(self):
        forest = OnlineForestRegressor(n_threads=123)
        self.assertEqual(forest._forest.n_threads(), 123)
        forest.n_threads = -12
        self.assertEqual(forest._forest.n_threads(), -12)
        # TODO: do it post fit

    def test_online_forest_regressor_n_trees(self):
        forest = OnlineForestRegressor(n_trees=123)
        self.assertEqual(forest._forest.n_trees(), 123)
        forest.n_trees = -12
        self.assertEqual(forest._forest.n_trees(), -12)
        # TODO: check that post-fit it cannot be changed

    def test_online_forest_regressor_criterion(self):
        forest = OnlineForestRegressor(criterion='mse')
        self.assertEqual(forest.criterion, 'mse')
        self.assertEqual(forest._forest.criterion(), mse)
        forest.criterion = 'unif'
        self.assertEqual(forest.criterion, 'unif')
        self.assertEqual(forest._forest.criterion(), unif)

        # msg = "^``criterion`` must be either 'unif' or 'mse'.$"
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     forest = OnlineForestRegressor(criterion='toto')
        # with self.assertRaisesRegex(RuntimeError, msg):
        #     forest.criterion = 123
        # TODO: check that post-fit it cannot be changed

    def test_online_forest_regressor_max_depth(self):
        forest = OnlineForestRegressor(max_depth=123)
        self.assertEqual(forest._forest.max_depth(), 123)
        forest.max_depth = -12
        self.assertEqual(forest._forest.max_depth(), -12)
        # TODO: check that post-fit it cannot be changed

    def test_online_forest_regressor_min_samples_split(self):
        forest = OnlineForestRegressor(min_samples_split=123)
        self.assertEqual(forest._forest.min_samples_split(), 123)
        forest.min_samples_split = 12
        self.assertEqual(forest._forest.min_samples_split(), 12)

    def test_online_forest_regressor_seed(self):
        forest = OnlineForestRegressor(seed=123)
        self.assertEqual(forest._forest.seed(), 123)
        forest.seed = -12
        self.assertEqual(forest._forest.seed(), -12)

    def test_online_forest_regressor_verbose(self):
        forest = OnlineForestRegressor(verbose=False)
        self.assertEqual(forest._forest.verbose(), False)
        forest.verbose = True
        self.assertEqual(forest._forest.verbose(), True)

    def test_online_forest_regressor_warm_start(self):
        forest = OnlineForestRegressor(warm_start=False)
        self.assertEqual(forest._forest.warm_start(), False)
        forest.warm_start = True
        self.assertEqual(forest._forest.warm_start(), True)

    def test_online_forest_regressor_n_splits(self):
        forest = OnlineForestRegressor(n_splits=False)
        self.assertEqual(forest._forest.n_splits(), False)
        forest.n_splits = True
        self.assertEqual(forest._forest.n_splits(), True)
        # TODO: n_splits must be positive

        # def test_ModelHuber(self):
        #     """...Numerical consistency check of loss and gradient for Huber model
        #     """
        #     np.random.seed(12)
        #     n_samples, n_features = 5000, 10
        #     w0 = np.random.randn(n_features)
        #     c0 = np.random.randn()
        #
        #     # First check with intercept
        #     X, y = SimuLinReg(w0, c0, n_samples=n_samples,
        #                       verbose=False).simulate()
        #     X_spars = csr_matrix(X)
        #     model = ModelHuber(fit_intercept=True, threshold=1.3).fit(X, y)
        #     model_spars = ModelHuber(fit_intercept=True,
        #                              threshold=1.3).fit(X_spars, y)
        #     self.run_test_for_glm(model, model_spars, 1e-5, 1e-3)
        #     self._test_glm_intercept_vs_hardcoded_intercept(model)
        #
        #     # Then check without intercept
        #     X, y = SimuLinReg(w0, None, n_samples=n_samples,
        #                       verbose=False, seed=2038).simulate()
        #     X_spars = csr_matrix(X)
        #     model = ModelHuber(fit_intercept=False).fit(X, y)
        #
        #     model_spars = ModelHuber(fit_intercept=False).fit(X_spars, y)
        #     self.run_test_for_glm(model, model_spars, 1e-5, 1e-3)
        #
        #     # Test for the Lipschitz constants without intercept
        #     self.assertAlmostEqual(model.get_lip_best(), 2.6873683857125981)
        #     self.assertAlmostEqual(model.get_lip_mean(), 9.95845726788432)
        #     self.assertAlmostEqual(model.get_lip_max(), 54.82616964855237)
        #     self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        #     self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())
        #
        #     # Test for the Lipschitz constants with intercept
        #     model = ModelHuber(fit_intercept=True).fit(X, y)
        #     model_spars = ModelHuber(fit_intercept=True).fit(X_spars, y)
        #     self.assertAlmostEqual(model.get_lip_best(), 2.687568385712598)
        #     self.assertAlmostEqual(model.get_lip_mean(), 10.958457267884327)
        #     self.assertAlmostEqual(model.get_lip_max(), 55.82616964855237)
        #     self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        #     self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

        # def test_ModelHuber_threshold(self):
        #     np.random.seed(12)
        #     n_samples, n_features = 5000, 10
        #     w0 = np.random.randn(n_features)
        #     c0 = np.random.randn()
        #     # First check with intercept
        #     X, y = SimuLinReg(w0, c0, n_samples=n_samples,
        #                       verbose=False).simulate()
        #
        #     model = ModelHuber(threshold=1.541).fit(X, y)
        #     self.assertEqual(model._model.get_threshold(), 1.541)
        #     model.threshold = 3.14
        #     self.assertEqual(model._model.get_threshold(), 3.14)
        #
        #     msg = '^threshold must be > 0$'
        #     with self.assertRaisesRegex(RuntimeError, msg):
        #         model = ModelHuber(threshold=-1).fit(X, y)
        #     with self.assertRaisesRegex(RuntimeError, msg):
        #         model.threshold = 0.


if __name__ == '__main__':
    unittest.main()
