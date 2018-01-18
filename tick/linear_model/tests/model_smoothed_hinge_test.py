# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLogReg, ModelSmoothedHinge
from tick.base_model.tests.generalized_linear_model import TestGLM


class Test(TestGLM):
    def test_ModelSmoothedHinge(self):
        """...Numerical consistency check of loss and gradient for SmoothedHinge
         model
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLogReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()
        X_spars = csr_matrix(X)
        model = ModelSmoothedHinge(fit_intercept=True, smoothness=0.2).fit(X, y)
        model_spars = ModelSmoothedHinge(fit_intercept=True,
                                         smoothness=0.2).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars, 1e-5, 1e-4)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLogReg(w0, None, n_samples=n_samples,
                          verbose=False, seed=2038).simulate()
        X_spars = csr_matrix(X)
        model = ModelSmoothedHinge(fit_intercept=False).fit(X, y)

        model_spars = ModelSmoothedHinge(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars, 1e-5, 1e-4)

        model = ModelSmoothedHinge(fit_intercept=False,
                                   smoothness=0.2).fit(X, y)
        model_spars = ModelSmoothedHinge(fit_intercept=False,
                                         smoothness=0.2).fit(X_spars, y)
        # Test for the Lipschitz constants without intercept
        self.assertAlmostEqual(model.get_lip_best(), 5 * 2.6873683857125981)
        self.assertAlmostEqual(model.get_lip_mean(), 5 * 9.95845726788432)
        self.assertAlmostEqual(model.get_lip_max(), 5 * 54.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

        # Test for the Lipschitz constants with intercept
        model = ModelSmoothedHinge(fit_intercept=True,
                                   smoothness=0.2).fit(X, y)
        model_spars = ModelSmoothedHinge(fit_intercept=True,
                                         smoothness=0.2).fit(X_spars, y)
        self.assertAlmostEqual(model.get_lip_best(), 5 * 2.687568385712598)
        self.assertAlmostEqual(model.get_lip_mean(), 5 * 10.958457267884327)
        self.assertAlmostEqual(model.get_lip_max(), 5 * 55.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

    def test_ModelSmoothedHinge_smoothness(self):
        np.random.seed(12)
        n_samples, n_features = 50, 2
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()
        # First check with intercept
        X, y = SimuLogReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()

        model = ModelSmoothedHinge(smoothness=0.123).fit(X, y)
        self.assertEqual(model._model.get_smoothness(), 0.123)
        model.smoothness = 0.765
        self.assertEqual(model._model.get_smoothness(), 0.765)

        msg = '^smoothness should be between 0.01 and 1$'
        with self.assertRaisesRegex(RuntimeError, msg):
            model = ModelSmoothedHinge(smoothness=-1).fit(X, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            model = ModelSmoothedHinge(smoothness=1.2).fit(X, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            model = ModelSmoothedHinge(smoothness=0.).fit(X, y)

        with self.assertRaisesRegex(RuntimeError, msg):
            model.smoothness = 0.
        with self.assertRaisesRegex(RuntimeError, msg):
            model.smoothness = -1.
        with self.assertRaisesRegex(RuntimeError, msg):
            model.smoothness = 2.


if __name__ == '__main__':
    unittest.main()
