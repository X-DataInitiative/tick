import unittest

import numpy as np
from scipy.sparse import csr_matrix

from tick.optim.model import ModelLinRegWithIntercepts
from tick.optim.model.tests.generalized_linear_model import TestGLM
from tick.simulation import SimuLinReg
from tick.simulation.base.weights import weights_sparse_gauss


class Test(TestGLM):
    def test_ModelLinRegWithIntercepts(self):
        """...Numerical consistency check of loss and gradient for Linear
        Regression
        """

        np.random.seed(12)
        n_samples, n_features = 200, 5
        w0 = np.random.randn(n_features)
        intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
        X, y = SimuLinReg(w0, None, n_samples=n_samples,
                          verbose=False, seed=2038).simulate()
        # Add gross outliers to the labels
        y += intercept0
        X_spars = csr_matrix(X)
        model = ModelLinRegWithIntercepts().fit(X, y)
        model_spars = ModelLinRegWithIntercepts().fit(X_spars, y)
        self.run_test_for_glm(model, model_spars, 1e-4, 1e-4)

        self.assertAlmostEqual(model.get_lip_mean(), 6.324960325598532)
        self.assertAlmostEqual(model.get_lip_max(), 30.277118951892113)
        self.assertAlmostEqual(model.get_lip_mean(), model_spars.get_lip_mean())
        self.assertAlmostEqual(model.get_lip_max(), model_spars.get_lip_max())

if __name__ == '__main__':
    unittest.main()
