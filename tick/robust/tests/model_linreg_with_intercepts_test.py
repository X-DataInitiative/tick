# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuLinReg
from tick.base_model.tests.generalized_linear_model import TestGLM
from tick.robust import ModelLinRegWithIntercepts
from tick.simulation import weights_sparse_gauss


class Test(TestGLM):
    def test_ModelLinRegWithInterceptsWithGlobalIntercept(self):
        """...Numerical consistency check of loss and gradient for linear
        regression with sample intercepts and a global intercept
        """
        np.random.seed(12)
        n_samples, n_features = 200, 5
        w0 = np.random.randn(n_features)
        intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
        c0 = None
        X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        # Add gross outliers to the labels
        y += intercept0
        X_spars = csr_matrix(X)
        model = ModelLinRegWithIntercepts(fit_intercept=False).fit(X, y)
        model_spars = ModelLinRegWithIntercepts(fit_intercept=False) \
            .fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        self.assertAlmostEqual(model.get_lip_mean(), 6.324960325598532)
        self.assertAlmostEqual(model.get_lip_max(), 30.277118951892113)
        self.assertAlmostEqual(model.get_lip_mean(),
                               model_spars.get_lip_mean())
        self.assertAlmostEqual(model.get_lip_max(), model_spars.get_lip_max())
        self.assertAlmostEqual(model.get_lip_best(), 2.7217793249045439)

    def test_ModelLinRegWithInterceptsWithoutGlobalIntercept(self):
        """...Numerical consistency check of loss and gradient for linear
        regression with sample intercepts and no global intercept
        """
        np.random.seed(12)
        n_samples, n_features = 200, 5
        w0 = np.random.randn(n_features)
        intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
        c0 = None
        X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        # Add gross outliers to the labels
        y += intercept0
        X_spars = csr_matrix(X)
        model = ModelLinRegWithIntercepts(fit_intercept=True).fit(X, y)
        model_spars = ModelLinRegWithIntercepts(fit_intercept=True) \
            .fit(X_spars, y)
        self.run_test_for_glm(model, model_spars)

        self.assertAlmostEqual(model.get_lip_mean(), 7.324960325598536)
        self.assertAlmostEqual(model.get_lip_max(), 31.277118951892113)
        self.assertAlmostEqual(model.get_lip_mean(),
                               model_spars.get_lip_mean())
        self.assertAlmostEqual(model.get_lip_max(), model_spars.get_lip_max())
        self.assertAlmostEqual(model.get_lip_best(), 2.7267793249045438)

    def test_ModelLinRegWithInterceptsWithoutGlobalInterceptExtras(self):
        """...Extra tests for linear regression with sample intercepts and not
        global intercept, check gradient wrt homemade gradient
        """
        np.random.seed(12)
        n_samples, n_features = 200, 5
        w0 = np.random.randn(n_features)
        intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
        c0 = None
        X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        # Add gross outliers to the labels
        y += intercept0
        model = ModelLinRegWithIntercepts(fit_intercept=False).fit(X, y)
        coeffs = np.random.randn(model.n_coeffs)
        grad1 = model.grad(coeffs)
        X2 = np.hstack((X, np.identity(n_samples)))
        grad2 = X2.T.dot(X2.dot(coeffs) - y) / n_samples
        np.testing.assert_almost_equal(grad1, grad2, decimal=10)

    def test_ModelLinRegWithInterceptsWithGlobalInterceptExtras(self):
        """...Extra tests for linear regression with sample intercepts and
        global intercept, check gradient wrt homemade gradient
        """
        np.random.seed(12)
        n_samples, n_features = 200, 5
        w0 = np.random.randn(n_features)
        intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
        c0 = -1.
        X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                          seed=2038).simulate()
        # Add gross outliers to the labels
        y += intercept0
        model = ModelLinRegWithIntercepts(fit_intercept=True).fit(X, y)
        coeffs = np.random.randn(model.n_coeffs)
        grad1 = model.grad(coeffs)
        X2 = np.hstack((X, np.ones((n_samples, 1)), np.identity(n_samples)))
        grad2 = X2.T.dot(X2.dot(coeffs) - y) / n_samples
        np.testing.assert_almost_equal(grad1, grad2, decimal=10)


if __name__ == '__main__':
    unittest.main()
