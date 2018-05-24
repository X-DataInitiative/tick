# License: BSD 3 clause

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.linear_model import SimuPoisReg, ModelPoisReg
from tick.base_model.tests.generalized_linear_model import TestGLM


class ModelPoisRegTest(object):
    def test_ModelPoisReg(self):
        """...Numerical consistency check of loss and gradient for Poisson
        Regression
        """

        np.random.seed(12)
        n_samples, n_features = 100, 10
        w0 = np.random.randn(n_features) / n_features
        c0 = np.random.randn() / n_features

        # First check with intercept
        X, y = SimuPoisReg(w0, c0, n_samples=n_samples, verbose=False,
                           seed=1234, dtype=self.dtype).simulate()
        # Rescale features since ModelPoisReg with exponential link
        #   (default) is not overflow proof
        X /= n_features
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelPoisReg(fit_intercept=True).fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=True).fit(X_spars, y)
        self.run_test_for_glm(model, model_sparse)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Then check without intercept
        X, y = SimuPoisReg(w0, None, n_samples=n_samples, verbose=False,
                           seed=1234, dtype=self.dtype).simulate()
        X /= n_features
        X_spars = csr_matrix(X, dtype=self.dtype)
        model = ModelPoisReg(fit_intercept=False).fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=False).fit(X_spars, y)
        self.run_test_for_glm(model, model_sparse)
        self._test_glm_intercept_vs_hardcoded_intercept(model)

        # Test the self-concordance constant
        n_samples, n_features = 5, 2
        X = np.zeros((n_samples, n_features))
        X_spars = csr_matrix(X)
        y = np.array([0, 0, 3, 2, 5], dtype=np.double)
        model = ModelPoisReg(fit_intercept=True, link="identity").fit(X, y)
        model_sparse = ModelPoisReg(fit_intercept=True, link="identity").fit(
            X_spars, y)
        self.assertAlmostEqual(model._sc_constant, 1.41421356237,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_sparse._sc_constant, 1.41421356237,
                               places=self.decimal_places)
        y = np.array([0, 0, 3, 2, 1], dtype=np.double)
        model.fit(X, y)
        model_sparse.fit(X_spars, y)
        self.assertAlmostEqual(model._sc_constant, 2.,
                               places=self.decimal_places)
        self.assertAlmostEqual(model_sparse._sc_constant, 2.,
                               places=self.decimal_places)

    def test_poisreg_sdca_rand_max(self):
        """...Test that SDCA's rand_max is worrect depending on link type
        """
        labels = np.array([0, 1, 2, 0, 4], dtype=self.dtype)
        features = np.random.rand(len(labels), 3).astype(self.dtype)

        model = ModelPoisReg(link='exponential').fit(features, labels)
        with self.assertRaises(NotImplementedError):
            model._sdca_rand_max
        self.assertEqual(model._rand_max, 5)

        model = ModelPoisReg(link='identity').fit(features, labels)
        self.assertEqual(model._sdca_rand_max, 3)
        self.assertEqual(model._rand_max, 5)


class ModelPoisRegTestFloat32(TestGLM, ModelPoisRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float32", **kwargs)


class ModelPoisRegTestFloat64(TestGLM, ModelPoisRegTest):
    def __init__(self, *args, **kwargs):
        TestGLM.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
