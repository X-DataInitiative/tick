# License: BSD 3 clause

import unittest

import numpy as np
from scipy.optimize import check_grad
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
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuPoisReg(w0, None, n_samples=n_samples,
                           verbose=False, seed=1234).simulate()
        X /= n_features
        X_spars = csr_matrix(X)
        model = ModelPoisReg(fit_intercept=False).fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_sparse, 1e-3, 1e-4)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

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

    def test_poisreg_sdca_rand_max(self):
        """...Test that SDCA's rand_max is worrect depending on link type
        """
        labels = np.array([0, 1, 2, 0, 4], dtype=float)
        features = np.random.rand(len(labels), 3)

        model = ModelPoisReg(link='exponential').fit(features, labels)
        with self.assertRaises(NotImplementedError):
            model._sdca_rand_max
        self.assertEqual(model._rand_max, 5)

        model = ModelPoisReg(link='identity').fit(features, labels)
        self.assertEqual(model._sdca_rand_max, 3)
        self.assertEqual(model._rand_max, 5)

    def test_poisreg_hessian(self):
        """...Numerical consistency check of hessian for Poisson regression
        """
        np.random.seed(19)
        n_samples, n_features = 100, 3
        w0 = np.random.randn(n_features) / n_features
        c0 = np.random.randn() / n_features
        X, y = SimuPoisReg(w0, c0, n_samples=n_samples,
                           verbose=False, seed=1234).simulate()

        for fit_intercept in [True, False]:
            model = ModelPoisReg(link='identity', fit_intercept=fit_intercept)
            model.fit(X, y)
            coeffs = np.random.rand(model.n_coeffs)
            hessian = model.hessian(coeffs)
            # Check that hessian is equal to its transpose
            np.testing.assert_array_almost_equal(hessian, hessian.T, decimal=9)

            np.set_printoptions(precision=3, linewidth=200)

            # Check that for all dimension hessian row is consistent
            # with its corresponding gradient coordinate.
            for i in range(model.n_coeffs):
                def g_i(x):
                    return model.grad(x)[i]

                def h_i(x):
                    h = model.hessian(x)
                    return np.asarray(h)[i, :]

                self.assertLess(check_grad(g_i, h_i, coeffs), 1e-3)


if __name__ == '__main__':
    unittest.main()
