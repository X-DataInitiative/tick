import unittest
import numpy as np

from tick.optim.model import ModelLogReg
from tick.optim.prox import ProxL1, ProxElasticNet
from tick.optim.solver import SDCA, SVRG
from tick.optim.solver.tests.solver import TestSolver


class Test(TestSolver):
    def test_solver_sdca(self):
        """...Check SDCA solver for a Logistic regression with Ridge
        penalization and L1 penalization
        """
        solver = SDCA(l_l2sq=1e-5, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)

        # Now a specific test with a real prox for SDCA
        np.random.seed(12)
        n_samples = Test.n_samples
        n_features = Test.n_features

        for fit_intercept in [True, False]:
            y, X, coeffs0, interc0 = TestSolver.generate_logistic_data(
                n_features, n_samples)

            model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
            ratio = 0.5
            l_enet = 1e-2

            # SDCA "elastic-net" formulation is different from elastic-net
            # implementation
            l_l2_sdca = ratio * l_enet
            l_l1_sdca = (1 - ratio) * l_enet
            sdca = SDCA(l_l2sq=l_l2_sdca, max_iter=100, verbose=False, tol=0,
                        seed=Test.sto_seed).set_model(model)
            prox_l1 = ProxL1(l_l1_sdca)
            sdca.set_prox(prox_l1)
            coeffs_sdca = sdca.solve()

            # Compare with SVRG
            svrg = SVRG(max_iter=100, verbose=False, tol=0,
                        seed=Test.sto_seed).set_model(model)
            prox_enet = ProxElasticNet(l_enet, ratio)
            svrg.set_prox(prox_enet)
            coeffs_svrg = svrg.solve(step=0.1)

            np.testing.assert_allclose(coeffs_sdca, coeffs_svrg)

    def test_sdca_sparse_and_dense_consistency(self):
        """...Test SDCA can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SDCA(max_iter=1, verbose=False, l_l2sq=1e-3,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)


if __name__ == '__main__':
    unittest.main()
