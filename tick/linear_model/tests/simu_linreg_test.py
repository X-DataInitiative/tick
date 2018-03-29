# License: BSD 3 clause

import unittest
import numpy as np

from tick.linear_model import SimuLinReg


class Test(unittest.TestCase):
    def test_SimuLinReg(self):
        """...Test simulation of a Linear Regression
        """
        n_samples = 10
        n_features = 3
        idx = np.arange(n_features)

        weights = np.exp(-idx / 10.)
        weights[::2] *= -1
        seed = 123
        simu = SimuLinReg(weights, None, n_samples=n_samples, seed=seed,
                          verbose=False)
        X, y = simu.simulate()

        X_truth = np.array([[1.4912667, 0.80881799, 0.26977298], [
            1.23227551, 0.50697013, 1.9409132
        ], [1.8891494, 1.49834791,
            2.41445794], [0.19431319, 0.80245126, 1.02577552], [
                -1.61687582, -1.08411865, -0.83438387
            ], [2.30419894, -0.68987056,
                -0.39750262], [-0.28826405, -1.23635074, -0.76124386],
                            [-1.32869473, -1.8752391,
                             -0.182537], [0.79464218, 0.65055633, 1.57572506],
                            [0.71524202, 1.66759831, 0.88679047]])

        y_truth = np.array([
            -1.23590872, -5.1612244, -4.28171221, -1.00793646, 2.24652287,
            -2.7766077, -0.20433269, 0.46957959, -2.37562537, 0.35124802
        ])

        np.testing.assert_array_almost_equal(X_truth, X)
        np.testing.assert_array_almost_equal(y_truth, y)


if __name__ == '__main__':
    unittest.main()
