# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes import ModelHawkesExpKernLogLik, SimuHawkesExpKernels
from tick.prox import ProxL2Sq
from tick.solver import SCPG
from tick.solver.tests.solver import TestSolver


class Test(TestSolver):
    def test_solver_scpg(self):
        """...Check Self-concordant proximal gradient solver for a Hawkes
        model with ridge penalization
        """
        beta = 3
        betas = beta * np.ones((2, 2))

        alphas = np.zeros((2, 2))

        alphas[0, 0] = 1
        alphas[0, 1] = 2
        alphas[1, 1] = 3

        mus = np.arange(1, 3) / 3

        hawkes = SimuHawkesExpKernels(adjacency=alphas, decays=betas,
                                      baseline=mus, seed=1231, end_time=20000,
                                      verbose=False)
        hawkes.adjust_spectral_radius(0.8)
        alphas = hawkes.adjacency

        hawkes.simulate()
        timestamps = hawkes.timestamps

        model = ModelHawkesExpKernLogLik(beta).fit(timestamps)
        prox = ProxL2Sq(1e-7, positive=True)
        pg = SCPG(max_iter=2000, tol=1e-10, verbose=False,
                  step=1e-5).set_model(model).set_prox(prox)

        pg.solve(np.ones(model.n_coeffs))

        original_coeffs = np.hstack((mus, alphas.reshape(4)))
        np.testing.assert_array_almost_equal(pg.solution, original_coeffs,
                                             decimal=2)


if __name__ == '__main__':
    unittest.main()
