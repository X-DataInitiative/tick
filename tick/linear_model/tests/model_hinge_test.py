# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLogReg, ModelHinge
from tick.base_model.tests.generalized_linear_model import TestGLM


class ModelHingeTest(object):
    def test_ModelHinge(self):
        """...Numerical consistency check of loss and gradient for Hinge model
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLogReg(w0, c0, n_samples=n_samples, verbose=False,
                          dtype=self.dtype).simulate()
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelHinge(fit_intercept=True).fit(X, y)
        model_spars = ModelHinge(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLogReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038, dtype=self.dtype).simulate()
        X_spars = csr_matrix(X)
        model = ModelHinge(fit_intercept=False).fit(X, y)

        model_spars = ModelHinge(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)


class ModelHingeTestFloat32(TestGLM, ModelHingeTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float32", **kwargs)


class ModelHingeTestFloat64(TestGLM, ModelHingeTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
