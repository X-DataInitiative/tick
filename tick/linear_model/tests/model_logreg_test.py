# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLogReg, ModelLogReg
from tick.base_model.tests.generalized_linear_model import TestGLM


class ModelLogRegTest(object):
    def test_ModelLogReg(self):
        """...Numerical consistency check of loss and gradient for Logistic
        Regression
        """

        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLogReg(w0, c0, n_samples=n_samples, verbose=False,
                          dtype=self.dtype).simulate()
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelLogReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLogReg(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLogReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038, dtype=self.dtype).simulate()
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelLogReg(fit_intercept=False).fit(X, y)

        model_spars = ModelLogReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Test for the Lipschitz constants without intercept
        self.assertAlmostEqual(model.get_lip_best(), 0.67184209642814952,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_mean(), 2.48961431697108,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_max(), 13.706542412138093,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean(),
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max(),
                               places=self.decimal_places)

        # Test for the Lipschitz constants with intercept
        model = ModelLogReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLogReg(fit_intercept=True).fit(X_spars, y)
        self.assertAlmostEqual(model.get_lip_best(), 0.671892096428,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_mean(), 2.739614316971082,
                               places=self.decimal_places)
        self.assertAlmostEqual(model.get_lip_max(), 13.956542412138093,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_mean(),
                               model.get_lip_mean(),
                               places=self.decimal_places)
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max(),
                               places=self.decimal_places)


class ModelLogRegTestFloat32(TestGLM, ModelLogRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float32", **kwargs)


class ModelLogRegTestFloat64(TestGLM, ModelLogRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
