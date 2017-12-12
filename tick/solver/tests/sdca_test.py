# License: BSD 3 clause

import unittest
import numpy as np

from tick.linear_model import ModelLogReg, ModelPoisReg, SimuPoisReg
from tick.prox import ProxL1, ProxElasticNet, ProxZero, ProxL2Sq
from tick.solver import SDCA, SVRG
from . import TestSolver


class Test(TestSolver):
    def test_solver_sdca(self):
        """...Check SDCA solver for a Logistic regression with Ridge
        penalization and L1 penalization
        """
        solver = SDCA(l_l2sq=1e-5, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)

    def compare_solver_sdca(self):
        """...Compare SDCA solution with SVRG solution
        """
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

    def test_sdca_identity_poisreg(self):
        """...Test SDCA on specific case of Poisson regression with
        indentity link
        """
        l_l2sq = 1e-3
        n_samples = 10000
        n_features = 3

        np.random.seed(123)
        weight0 = np.random.rand(n_features)
        features = np.random.rand(n_samples, n_features)

        for intercept in [None, 0.45]:
            if intercept is None:
                fit_intercept = False
            else:
                fit_intercept = True

            simu = SimuPoisReg(weight0, intercept=intercept,
                               features=features, n_samples=n_samples,
                               link='identity', verbose=False)
            features, labels = simu.simulate()

            model = ModelPoisReg(fit_intercept=fit_intercept, link='identity')
            model.fit(features, labels)

            sdca = SDCA(l_l2sq=l_l2sq, max_iter=100, verbose=False,
                        tol=1e-14, seed=Test.sto_seed)

            sdca.set_model(model).set_prox(ProxZero())
            start_dual = np.sqrt(sdca._rand_max * l_l2sq)
            start_dual = start_dual * np.ones(sdca._rand_max)

            sdca.solve(start_dual)

            # Check that duality gap is 0
            self.assertAlmostEqual(sdca.objective(sdca.solution),
                                   sdca.dual_objective(sdca.dual_solution))

            # Check that original vector is approximatively retrieved
            if fit_intercept:
                original_coeffs = np.hstack((weight0, intercept))
            else:
                original_coeffs = weight0

            np.testing.assert_array_almost_equal(original_coeffs, sdca.solution,
                                                 decimal=1)

            # Ensure that we solve the same problem as other solvers
            svrg = SVRG(max_iter=100, verbose=False,
                        tol=1e-14, seed=Test.sto_seed)

            svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq))
            svrg.solve(0.5 * np.ones(model.n_coeffs), step=1e-2)
            np.testing.assert_array_almost_equal(svrg.solution, sdca.solution,
                                                 decimal=4)


if __name__ == '__main__':
    unittest.main()
