import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.optim.model import ModelPoisReg
from tick.optim.model.tests.generalized_linear_model import TestGLM
from tick.simulation import SimuPoisReg


class Test(TestGLM):
    def test_ModelPoisReg(self):
        """...Numerical consistency check of loss and gradient for Poisson
        Regression
        """

        np.random.seed(12)
        n_samples, n_features = 100, 10
        w0 = np.random.randn(n_features) / n_features
        c0 = np.random.randn() / n_features

        # First check with intercept
        X, y = SimuPoisReg(w0, c0, n_samples=n_samples,
                           verbose=False, seed=1234).simulate()
        # Rescale features since ModelPoisReg with exponential link
        #   (default) is not overflow proof
        X /= n_features
        X_spars = csr_matrix(X)
        model = ModelPoisReg(fit_intercept=True).fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_sparse, 1e-3, 1e-4)

        # Then check without intercept
        X, y = SimuPoisReg(w0, None, n_samples=n_samples,
                           verbose=False, seed=1234).simulate()
        X /= n_features
        X_spars = csr_matrix(X)
        model = ModelPoisReg(fit_intercept=False).fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_sparse, 1e-3, 1e-4)

        # Test the self-concordance constant
        n_samples, n_features = 5, 2
        X = np.zeros((n_samples, n_features))
        X_spars = csr_matrix(X)
        y = np.array([0, 0, 3, 2, 5], dtype=np.double)
        model = ModelPoisReg(fit_intercept=True,
                             link="identity").fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=True,
                                    link="identity").fit(X_spars, y)
        self.assertAlmostEqual(model._sc_constant, 1.41421356237)
        self.assertAlmostEqual(model_sparse._sc_constant, 1.41421356237)
        y = np.array([0, 0, 3, 2, 1], dtype=np.double)
        model.fit(X, y)
        model_sparse.fit(X_spars, y)
        self.assertAlmostEqual(model._sc_constant, 2.)
        self.assertAlmostEqual(model_sparse._sc_constant, 2.)


if __name__ == '__main__':
    unittest.main()
