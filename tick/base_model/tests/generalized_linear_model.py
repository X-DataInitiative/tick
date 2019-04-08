# License: BSD 3 clause
import warnings

import numpy as np
from numpy.linalg import norm
from scipy.optimize import check_grad, fmin_bfgs

import unittest


class TestGLM(unittest.TestCase):
    def __init__(self, *args, dtype="float64", **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.dtype = dtype
        self.decimal_places = 7
        self.delta_check_grad = 1e-5
        self.delta_model_grad = 2e-3
        if np.dtype(self.dtype) == np.dtype("float32"):
            self.decimal_places = 3
            self.delta_check_grad = 3e-2
            self.delta_model_grad = 2e-2

    def _test_grad(self, model, coeffs, delta_check_grad=None,
                   delta_model_grad=None):
        """Test that gradient is consistent with loss and that minimum is
        achievable with a small gradient
        """
        if coeffs.dtype is np.dtype("float32"):
            check_grad_epsilon = 3e-3
        else:
            check_grad_epsilon = 1e-7

        if delta_check_grad is None:
            delta_check_grad = self.delta_check_grad

        if delta_model_grad is None:
            delta_model_grad = self.delta_model_grad

        with warnings.catch_warnings(record=True):
            grad_check = check_grad(model.loss, model.grad, coeffs,
                                    epsilon=check_grad_epsilon)

        self.assertAlmostEqual(grad_check, 0., delta=delta_check_grad)
        # Check that minimum is achievable with a small gradient

        with warnings.catch_warnings(record=True):
            coeffs_min = fmin_bfgs(model.loss, coeffs, fprime=model.grad,
                                   disp=False)
            coeffs_min = coeffs_min.astype(self.dtype)

        self.assertAlmostEqual(
            norm(model.grad(coeffs_min)), .0, delta=delta_model_grad)

    def run_test_for_glm(self, model, model_spars=None, delta_check_grad=None,
                         delta_model_grad=None):
        coeffs = np.random.randn(model.n_coeffs).astype(self.dtype)
        # dense case
        self._test_grad(model, coeffs, delta_check_grad=delta_check_grad,
                        delta_model_grad=delta_model_grad)
        # sparse case
        if model_spars is not None:
            self._test_grad(model_spars, coeffs,
                            delta_check_grad=delta_check_grad,
                            delta_model_grad=delta_model_grad)
            # Check that loss computed in the dense and sparse case are
            # the same
            self.assertAlmostEqual(
                model.loss(coeffs), model_spars.loss(coeffs),
                places=self.decimal_places)
            # Check that gradients computed in the dense and sparse
            # case are the same
            np.testing.assert_almost_equal(
                model.grad(coeffs), model_spars.grad(coeffs),
                decimal=self.decimal_places)

        if model.dtype != 'float64':
            model_float64 = model.astype('float64')

            self.assertAlmostEqual(
                model.loss(coeffs), model_float64.loss(coeffs.astype(float)),
                places=self.decimal_places)
            np.testing.assert_almost_equal(
                model.grad(coeffs), model_float64.grad(coeffs.astype(float)),
                decimal=self.decimal_places)

    def _test_glm_intercept_vs_hardcoded_intercept(self, model):
        # If the model has an intercept (ModelCoxReg does not for instance)
        if hasattr(model, 'fit_intercept'):
            # For the model with intercept only, test that
            if model.fit_intercept:
                X = model.features
                y = model.labels
                coeffs = np.random.randn(model.n_coeffs).astype(self.dtype)
                grad1 = model.grad(coeffs)

                X_with_ones = np.hstack((X, np.ones((model.n_samples,
                                                     1)))).astype(self.dtype)
                model.fit_intercept = False
                model.fit(X_with_ones, y)
                grad2 = model.grad(coeffs)

                np.testing.assert_almost_equal(grad1, grad2,
                                               decimal=self.decimal_places)

                # Put back model to its previous status
                model.fit_intercept = True
                model.fit(X, y)
