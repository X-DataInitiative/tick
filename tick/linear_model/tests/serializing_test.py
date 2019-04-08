# License: BSD 3 clause

import io, unittest
import numpy as np

import pickle
import scipy.sparse
from scipy.sparse import csr_matrix

from tick.solver.tests import TestSolver

from tick.prox import ProxL1
from tick.linear_model import ModelLinReg, SimuLinReg
from tick.linear_model import ModelLogReg, SimuLogReg
from tick.linear_model import ModelPoisReg, SimuPoisReg
from tick.linear_model import ModelHinge, ModelQuadraticHinge, ModelSmoothedHinge

from tick.simulation import weights_sparse_gauss


class Test(TestSolver):
    def test_linear_model_serialization(self):
        """...Test serialization of linear models
        """
        model_map = {
            ModelLinReg: SimuLinReg,
            ModelLogReg: SimuLogReg,
            ModelPoisReg: SimuPoisReg,
            ModelHinge: SimuLogReg,
            ModelQuadraticHinge: SimuLogReg,
            ModelSmoothedHinge: SimuLogReg,
        }

        for mod in model_map:
            model = mod(fit_intercept=False)

            coeffs0 = weights_sparse_gauss(20, nnz=5)
            interc0 = None

            features, labels = model_map[mod](coeffs0, interc0, n_samples=100,
                                              verbose=False,
                                              seed=123).simulate()
            model.fit(features, labels)

            pickled = pickle.loads(pickle.dumps(model))

            self.assertTrue(model._model.compare(pickled._model))
            self.assertEqual(
                model.loss(features[0]), pickled.loss(features[0]))

    def test_sparse_linear_model_serialization(self):
        """...Test serialization of linear models with sparse features
        """
        model_map = {
            ModelLinReg: SimuLinReg,
            ModelLogReg: SimuLogReg,
            ModelPoisReg: SimuPoisReg,
            ModelHinge: SimuLogReg,
            ModelQuadraticHinge: SimuLogReg,
            ModelSmoothedHinge: SimuLogReg,
        }

        for mod in model_map:
            model = mod(fit_intercept=False)

            coeffs0 = weights_sparse_gauss(20, nnz=5)
            interc0 = None
            features = scipy.sparse.random(100, len(coeffs0), format='csr')
            features, labels = model_map[mod](coeffs0, interc0, features,
                                              n_samples=100, verbose=False,
                                              seed=123).simulate()

            model.fit(features, labels)
            pickled = pickle.loads(pickle.dumps(model))

            self.assertTrue(model._model.compare(pickled._model))
            coeffs = np.random.rand(pickled.n_coeffs)
            self.assertEqual(model.loss(coeffs), pickled.loss(coeffs))

if __name__ == "__main__":
    unittest.main()
