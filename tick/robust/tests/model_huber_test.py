# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.robust import ModelHuber
from tick.base_model.tests.generalized_linear_model import TestGLM
from tick.linear_model import SimuLinReg


class Test(TestGLM):
    def test_ModelHuber(self):
        """...Numerical consistency check of loss and gradient for Huber model
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()
        X_spars = csr_matrix(X)
        model = ModelHuber(fit_intercept=True, threshold=1.3).fit(X, y)
        model_spars = ModelHuber(fit_intercept=True, threshold=1.3).fit(
            X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLinReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        X_spars = csr_matrix(X)
        model = ModelHuber(fit_intercept=False).fit(X, y)

        model_spars = ModelHuber(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)

        # Test for the Lipschitz constants without intercept
        self.assertAlmostEqual(model.get_lip_best(), 2.6873683857125981)
        self.assertAlmostEqual(model.get_lip_mean(), 9.95845726788432)
        self.assertAlmostEqual(model.get_lip_max(), 54.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

        # Test for the Lipschitz constants with intercept
        model = ModelHuber(fit_intercept=True).fit(X, y)
        model_spars = ModelHuber(fit_intercept=True).fit(X_spars, y)
        self.assertAlmostEqual(model.get_lip_best(), 2.687568385712598)
        self.assertAlmostEqual(model.get_lip_mean(), 10.958457267884327)
        self.assertAlmostEqual(model.get_lip_max(), 55.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

    def test_ModelHuber_threshold(self):
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()
        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()

        model = ModelHuber(threshold=1.541).fit(X, y)
        self.assertEqual(model._model.get_threshold(), 1.541)
        model.threshold = 3.14
        self.assertEqual(model._model.get_threshold(), 3.14)

        msg = '^threshold must be > 0$'
        with self.assertRaisesRegex(RuntimeError, msg):
            model = ModelHuber(threshold=-1).fit(X, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            model.threshold = 0.


if __name__ == '__main__':
    unittest.main()
