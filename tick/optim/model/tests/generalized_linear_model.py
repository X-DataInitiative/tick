import numpy as np
from numpy.linalg import norm
from scipy.optimize import check_grad, fmin_bfgs

import unittest


class TestGLM(unittest.TestCase):
    def _test_grad(self, model, coeffs,
                   delta_check_grad=1e-5,
                   delta_model_grad=1e-4):
        """Test that gradient is consistent with loss and that minimum is
        achievable with a small gradient
        """
        self.assertAlmostEqual(check_grad(model.loss,
                                          model.grad,
                                          coeffs),
                               0.,
                               delta=delta_check_grad)
        # Check that minimum iss achievable with a small gradient
        coeffs_min = fmin_bfgs(model.loss, coeffs,
                               fprime=model.grad, disp=False)
        self.assertAlmostEqual(norm(model.grad(coeffs_min)),
                               .0, delta=delta_model_grad)

    def run_test_for_glm(self, model, model_spars=None,
                         delta_check_grad=1e-5,
                         delta_model_grad=1e-4):
        """A generic test for generalized linear models (called by others)
        """
        coeffs = np.random.randn(model.n_coeffs)

        # dense case
        self._test_grad(model, coeffs,
                        delta_check_grad=delta_check_grad,
                        delta_model_grad=delta_model_grad)
        # sparse case
        if model_spars is not None:
            self._test_grad(model_spars, coeffs,
                            delta_check_grad=delta_check_grad,
                            delta_model_grad=delta_model_grad)

            # Check that loss computed in the dense and sparse case are
            # the same
            self.assertAlmostEqual(model.loss(coeffs),
                                   model_spars.loss(coeffs))
            # Check that gradients computed in the dense and sparse
            # case are the same
            np.testing.assert_almost_equal(model.grad(coeffs),
                                           model_spars.grad(coeffs),
                                           decimal=10)
