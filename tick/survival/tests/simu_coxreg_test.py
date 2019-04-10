# License: BSD 3 clause

import unittest
import numpy as np

from tick.survival import SimuCoxReg, SimuCoxRegWithCutPoints


class Test(unittest.TestCase):
    def test_SimuCoxReg(self):
        """...Test simulation of a Cox Regression
        """
        # Simulate a Cox model with specific seed
        n_samples = 10
        n_features = 3
        idx = np.arange(n_features)
        # Parameters of the Cox simu
        coeffs = np.exp(-idx / 10.)
        coeffs[::2] *= -1

        seed = 123
        simu = SimuCoxReg(coeffs, n_samples=n_samples, seed=seed,
                          verbose=False)
        features_, times_, censoring_ = simu.simulate()

        times = np.array([
            1.5022119, 5.93102441, 6.82837051, 0.50940341, 0.14859682,
            30.22922996, 3.54945974, 0.8671229, 1.4228358, 0.11483298
        ])

        censoring = np.array([1, 0, 1, 1, 1, 1, 1, 1, 0, 1], dtype=np.ushort)

        features = np.array([[1.4912667, 0.80881799, 0.26977298], [
            1.23227551, 0.50697013, 1.9409132
        ], [1.8891494, 1.49834791,
            2.41445794], [0.19431319, 0.80245126, 1.02577552], [
                                 -1.61687582, -1.08411865, -0.83438387
                             ], [2.30419894, -0.68987056,
                                 -0.39750262],
                             [-0.28826405, -1.23635074, -0.76124386], [
                                 -1.32869473, -1.8752391, -0.182537
                             ], [0.79464218, 0.65055633, 1.57572506],
                             [0.71524202, 1.66759831, 0.88679047]])

        np.testing.assert_almost_equal(features, features_)
        np.testing.assert_almost_equal(times, times_)
        np.testing.assert_almost_equal(censoring, censoring_)

    def test_SimuCoxRegWithCutPoints(self):
        """...Test simulation of a Cox Regression with cut-points
        """
        # Simulate a Cox model with cut-points with specific seed
        n_samples = 10
        n_features = 3
        n_cut_points = 2
        cov_corr = .5
        sparsity = .2

        seed = 123
        simu = SimuCoxRegWithCutPoints(n_samples=n_samples,
                                       n_features=n_features,
                                       seed=seed, verbose=False,
                                       n_cut_points=n_cut_points,
                                       shape=2, scale=.1, cov_corr=cov_corr,
                                       sparsity=sparsity)
        features_, times_, censoring_, cut_points_, coeffs_binarized_, S_ = simu.simulate()

        times = np.array([6.12215425, 6.74403919, 5.2148425, 5.42903238,
                          2.42953933, 9.50705158, 18.49545933, 19.7929349,
                          0.39267278, 1.24799812])

        censoring = np.array([1, 0, 0, 1, 0, 1, 1, 1, 0, 1], dtype=np.ushort)

        features = np.array([[1.4912667, 0.80881799, 0.26977298],
                             [1.23227551, 0.50697013, 1.9409132],
                             [1.8891494, 1.49834791, 2.41445794],
                             [0.19431319, 0.80245126, 1.02577552],
                             [-1.61687582, -1.08411865, -0.83438387],
                             [2.30419894, -0.68987056, -0.39750262],
                             [-0.28826405, -1.23635074, -0.76124386],
                             [-1.32869473, -1.8752391, -0.182537],
                             [0.79464218, 0.65055633, 1.57572506],
                             [0.71524202, 1.66759831, 0.88679047]])

        cut_points = {'0': np.array([-np.inf, -0.28826405, 0.79464218, np.inf]),
                      '1': np.array([-np.inf, -1.23635074, 0.50697013, np.inf]),
                      '2': np.array([-np.inf, -0.182537, 0.88679047, np.inf])}

        coeffs_binarized = np.array([-1.26789642, 1.31105319, -0.04315676, 0.,
                                     0., 0., 0.01839684, 0.4075832,
                                     -0.42598004])

        S = np.array([1])

        np.testing.assert_almost_equal(features, features_)
        np.testing.assert_almost_equal(times, times_)
        np.testing.assert_almost_equal(censoring, censoring_)
        np.testing.assert_almost_equal(cut_points_['0'], cut_points['0'])
        np.testing.assert_almost_equal(cut_points_['1'], cut_points['1'])
        np.testing.assert_almost_equal(cut_points_['2'], cut_points['2'])
        np.testing.assert_almost_equal(coeffs_binarized, coeffs_binarized_)
        np.testing.assert_almost_equal(S, S_)


if __name__ == '__main__':
    unittest.main()
