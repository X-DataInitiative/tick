# License: BSD 3 clause

import io, unittest
import numpy as np

import pickle
from scipy.sparse import csr_matrix

from tick.base_model.tests.generalized_linear_model import TestGLM

from tick.prox import ProxL1
from tick.linear_model import ModelLinReg, SimuLinReg
from tick.linear_model import ModelLogReg, SimuLogReg
from tick.linear_model import ModelPoisReg, SimuPoisReg
from tick.linear_model import ModelHinge, ModelQuadraticHinge, ModelSmoothedHinge

from tick.robust import ModelAbsoluteRegression, ModelEpsilonInsensitive, ModelHuber, \
                        ModelLinRegWithIntercepts, ModelModifiedHuber

from tick.simulation import weights_sparse_gauss


class Test(TestGLM):
    def test_robust_model_serialization(self):
        """...Test serialization of robust models
        """
        model_map = {
            ModelAbsoluteRegression: SimuLinReg,
            ModelEpsilonInsensitive: SimuLinReg,
            ModelHuber: SimuLinReg,
            ModelLinRegWithIntercepts: SimuLinReg,
            ModelModifiedHuber: SimuLogReg
        }

        for mod in model_map:
            np.random.seed(12)
            n_samples, n_features = 100, 5
            w0 = np.random.randn(n_features)
            intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples, nnz=30)
            c0 = None
            X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                              seed=2038).simulate()

            if mod == ModelLinRegWithIntercepts:
                y += intercept0

            model = mod(fit_intercept=False).fit(X, y)

            pickled = pickle.loads(pickle.dumps(model))

            self.assertTrue(model._model.compare(pickled._model))

            if mod == ModelLinRegWithIntercepts:
                test_vector = np.hstack((X[0], np.ones(n_samples)))
                self.assertEqual(
                    model.loss(test_vector), pickled.loss(test_vector))
            else:
                self.assertEqual(model.loss(X[0]), pickled.loss(X[0]))


if __name__ == "__main__":
    unittest.main()
