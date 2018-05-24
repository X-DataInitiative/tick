# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.robust import ModelEpsilonInsensitive
from tick.base_model.tests.generalized_linear_model import TestGLM
from tick.linear_model import SimuLinReg


class Test(TestGLM):
    def test_ModelEpsilonInsensitive(self):
        """...Numerical consistency check of loss and gradient for
        Epsilon-Insensitive model
        """
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()

        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()
        X_spars = csr_matrix(X)
        model = ModelEpsilonInsensitive(fit_intercept=True,
                                        threshold=1.13).fit(X, y)
        model_spars = ModelEpsilonInsensitive(fit_intercept=True,
                                              threshold=1.13).fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuLinReg(w0, None, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        X_spars = csr_matrix(X)
        model = ModelEpsilonInsensitive(fit_intercept=False).fit(X, y)
        model_spars = ModelEpsilonInsensitive(fit_intercept=False).fit(
            X_spars, y)
        self.run_test_for_glm(model, model_spars)

    def test_ModelEpsilonInsensitive_threshold(self):
        np.random.seed(12)
        n_samples, n_features = 5000, 10
        w0 = np.random.randn(n_features)
        c0 = np.random.randn()
        # First check with intercept
        X, y = SimuLinReg(w0, c0, n_samples=n_samples,
                          verbose=False).simulate()

        model = ModelEpsilonInsensitive(threshold=1.541).fit(X, y)
        self.assertEqual(model._model.get_threshold(), 1.541)
        model.threshold = 3.14
        self.assertEqual(model._model.get_threshold(), 3.14)

        msg = '^threshold must be > 0$'
        with self.assertRaisesRegex(RuntimeError, msg):
            model = ModelEpsilonInsensitive(threshold=-1).fit(X, y)
        with self.assertRaisesRegex(RuntimeError, msg):
            model.threshold = 0.


if __name__ == '__main__':
    unittest.main()
