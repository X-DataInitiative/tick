# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLinReg, ModelLinReg
from tick.base_model.tests.generalized_linear_model import TestGLM


class ModelLinRegTest(object):
    def test_ModelLinReg(self):
        """...Numerical consistency check of loss and gradient for Linear
        Regression
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features).astype(self.dtype)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                          dtype=self.dtype).simulate()
        X_spars = csr_matrix(X, dtype=self.dtype)

        model = ModelLinReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLinReg(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLinReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038, dtype=self.dtype).simulate()
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelLinReg(fit_intercept=False).fit(X, y)

        model_spars = ModelLinReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Test for the Lipschitz constants without intercept
        self.assertAlmostEqual(model.get_lip_best(), 2.6873683857125981,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_mean(), 9.95845726788432,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_max(), 54.82616964855237,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean(),
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max(),
                               places=self.decimal_places)

        # Test for the Lipschitz constants with intercept
        model = ModelLinReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLinReg(fit_intercept=True).fit(X_spars, y)
        self.assertAlmostEqual(model.get_lip_best(), 2.687568385712598,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_mean(), 10.958457267884327,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_max(), 55.82616964855237,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean(),
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max(),
                               places=self.decimal_places)


class ModelLinRegTestFloat32(TestGLM, ModelLinRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float32", **kwargs)


class ModelLinRegTestFloat64(TestGLM, ModelLinRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
