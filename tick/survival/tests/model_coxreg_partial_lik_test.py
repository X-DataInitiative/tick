# License: BSD 3 clause

import unittest, pickle

import numpy as np
from scipy.sparse import csr_matrix

from tick.survival import SimuCoxReg, ModelCoxRegPartialLik
from tick.base_model.tests.generalized_linear_model import TestGLM


class Test(TestGLM):
    def test_ModelCoxRegPartialLik(self):
        """...Numerical consistency check of loss and gradient for Cox Regression
        """
        np.random.seed(123)
        n_samples, n_features = 100, 5
        w0 = np.random.randn(n_features)
        features, times, censoring = SimuCoxReg(
            w0, n_samples=n_samples, verbose=False, seed=1234).simulate()
        model = ModelCoxRegPartialLik()
        model.fit(features, times, censoring)
        model_spars = ModelCoxRegPartialLik()
        model_spars.fit(csr_matrix(features), times, censoring)
        self.run_test_for_glm(model, model_spars)

    def test_coxreg_serialize_and_compare(self):
        """Test serialization (cereal/pickle) of Cox Regression."""
        np.random.seed(123)
        n_samples, n_features = 100, 5
        w0 = np.random.randn(n_features)
        features, times, censoring = SimuCoxReg(
            w0, n_samples=n_samples, verbose=False, seed=1234).simulate()

        model = ModelCoxRegPartialLik()
        model.fit(features, times, censoring)
        pickled = pickle.loads(pickle.dumps(model))
        self.assertTrue(model._model.compare(pickled._model))

if __name__ == '__main__':
    unittest.main()
