# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLinReg, ModelLinReg
from tick.base_model.tests.generalized_linear_model import TestGLM


class Test(TestGLM):
    def test_ModelLinReg(self):
        """...Numerical consistency check of loss and gradient for Linear
        Regression
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()
        X_spars = csr_matrix(X)
        model = ModelLinReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLinReg(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars, 1e-5, 1e-4)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLinReg(w0, None, n_samples=n_samples,
                          verbose=False, seed=2038).simulate()
        X_spars = csr_matrix(X)
        model = ModelLinReg(fit_intercept=False).fit(X, y)

        model_spars = ModelLinReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars, 1e-5, 1e-4)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Test for the Lipschitz constants without intercept
        self.assertAlmostEqual(model.get_lip_best(), 2.6873683857125981)
        self.assertAlmostEqual(model.get_lip_mean(), 9.95845726788432)
        self.assertAlmostEqual(model.get_lip_max(), 54.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())

        # Test for the Lipschitz constants with intercept
        model = ModelLinReg(fit_intercept=True).fit(X, y)
        model_spars = ModelLinReg(fit_intercept=True).fit(X_spars, y)
        self.assertAlmostEqual(model.get_lip_best(), 2.687568385712598)
        self.assertAlmostEqual(model.get_lip_mean(), 10.958457267884327)
        self.assertAlmostEqual(model.get_lip_max(), 55.82616964855237)
        self.assertAlmostEqual(model_spars.get_lip_mean(), model.get_lip_mean())
        self.assertAlmostEqual(model_spars.get_lip_max(), model.get_lip_max())


if __name__ == '__main__':
    unittest.main()
