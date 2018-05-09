# License: BSD 3 clause

import io, unittest
import numpy as np

import pickle

from tick.solver.tests import TestSolver

from tick.prox import ProxL1
from tick.linear_model import ModelLinReg, SimuLinReg
from tick.linear_model import ModelLogReg, SimuLogReg
from tick.linear_model import ModelPoisReg, SimuPoisReg
from tick.linear_model import ModelHinge, ModelQuadraticHinge, ModelSmoothedHinge

from tick.robust import ModelAbsoluteRegression, ModelEpsilonInsensitive, ModelHuber, \
                        ModelLinRegWithIntercepts, ModelModifiedHuber

from tick.solver import AdaGrad, SGD, SDCA, SAGA, SVRG

from tick.simulation import weights_sparse_gauss

class Test(TestSolver):
    def test_serializing_solvers(self):
        """...Test serialization of solvers
        """
        ratio = 0.5
        l_enet = 1e-2
        sd = ratio * l_enet

        solvers = [
            AdaGrad(step=1e-3, max_iter=100, verbose=False, tol=0),
            SGD(step=1e-3, max_iter=100, verbose=False, tol=0),
            SDCA(l_l2sq=sd, max_iter=100, verbose=False, tol=0),
            SAGA(step=1e-3, max_iter=100, verbose=False, tol=0),
            SVRG(step=1e-3, max_iter=100, verbose=False, tol=0)
        ]
        model_map = {
            ModelLinReg: SimuLinReg,
            ModelLogReg: SimuLogReg,
            ModelPoisReg: SimuPoisReg,
            ModelHinge: SimuLogReg,
            ModelQuadraticHinge: SimuLogReg,
            ModelSmoothedHinge: SimuLogReg,
            ModelAbsoluteRegression: SimuLinReg,
            ModelEpsilonInsensitive: SimuLinReg,
            ModelHuber: SimuLinReg,
            ModelLinRegWithIntercepts: SimuLinReg,
            ModelModifiedHuber: SimuLogReg
        }
        for solver in solvers:
            for mod in model_map:

                np.random.seed(12)
                n_samples, n_features = 100, 5
                w0 = np.random.randn(n_features)
                intercept0 = 50 * weights_sparse_gauss(n_weights=n_samples,
                                                       nnz=30, dtype=self.dtype)
                c0 = None
                X, y = SimuLinReg(w0, c0, n_samples=n_samples, verbose=False,
                                  seed=2038,dtype=self.dtype).simulate()

                if mod == ModelLinRegWithIntercepts:
                    y += intercept0

                model = mod(fit_intercept=False).fit(X, y)

                prox = ProxL1(2.)
                solver.set_model(model)
                solver.set_prox(prox)

                pickled = pickle.loads(pickle.dumps(solver))

                self.assertTrue(solver._solver.compare(pickled._solver))

                self.assertTrue(
                    solver.model._model.compare(pickled.model._model))

                self.assertTrue(solver.prox._prox.compare(pickled.prox._prox))

                if mod == ModelLinRegWithIntercepts:
                    test_vector = np.hstack((X[0], np.ones(n_samples)))
                    self.assertEqual(
                        model.loss(test_vector),
                        solver.model.loss(test_vector))
                else:
                    self.assertEqual(model.loss(X[0]), solver.model.loss(X[0]))

if __name__ == "__main__":
    suite = unittest.TestSuite()
    for dt in dtype_list:
        suite.addTest(parameterize(SolverTest, dtype=dt))
    unittest.TextTestRunner().run(suite)
