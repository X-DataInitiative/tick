# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.robust import ModelAbsoluteRegression
from tick.base_model.tests.generalized_linear_model import TestGLM
from tick.linear_model import SimuLinReg


class Test(TestGLM):
    def test_ModelAbsoluteRegression(self):
        """...Numerical consistency check of loss and gradient for Hinge model
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()
        X_spars = csr_matrix(X)
        model = ModelAbsoluteRegression(fit_intercept=True).fit(X, y)
        model_spars = ModelAbsoluteRegression(fit_intercept=True).fit(
            X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLinReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        X_spars = csr_matrix(X)
        model = ModelAbsoluteRegression(fit_intercept=False).fit(X, y)

        model_spars = ModelAbsoluteRegression(fit_intercept=False).fit(
            X_spars, y)
        self.run_test_for_glm(model, model_spars)


if __name__ == '__main__':
    unittest.main()
