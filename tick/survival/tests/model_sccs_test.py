# License: BSD 3 clause

import unittest, pickle
import numpy as np
from scipy.optimize import check_grad, fmin_bfgs
from scipy.linalg import norm
from scipy.sparse import csr_matrix
from tick.survival import SimuSCCS, ModelSCCS
from tick.preprocessing import LongitudinalFeaturesLagger
from tick.solver import SVRG
from tick.prox import ProxZero


class ModelSCCSTest(unittest.TestCase):
    def setUp(self):
        self.X = [
            np.array([[0, 1], [0, 1]], dtype="float64"),
            np.array([[1, 1], [1, 0]], dtype="float64")
        ]
        self.y = [
            np.array([1, 0], dtype="int32"),
            np.array([0, 1], dtype="int32")
        ]

        self.n_lags = np.array([1, 1], dtype="uint64")

    def test_loss(self):
        """Test longitudinal multinomial model loss."""
        X, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags)\
            .fit_transform(self.X)
        model = ModelSCCS(n_intervals=2, n_lags=self.n_lags)\
            .fit(X, self.y)
        loss = model.loss(coeffs=np.array([0.0, 0.0, 1.0, 0.0]))
        expected_loss = -np.log((np.e / (2 * np.e) * 1 / (1 + np.e))) / 2
        self.assertAlmostEqual(loss, expected_loss)

    def test_grad(self):
        """Test longitudinal multinomial model gradient value."""
        X = [np.array([[0, 0.], [1, 0]]), np.array([[1, 0.], [0, 1]])]
        X, _, _ = LongitudinalFeaturesLagger(n_lags=self.n_lags) \
            .fit_transform(X)
        model = ModelSCCS(n_intervals=2, n_lags=self.n_lags) \
            .fit(X, self.y)
        grad = model.grad(coeffs=np.array([0.0, 0.0, 1.0, 0.0]))
        expected_grad = -np.array([
            -1 / 2 - 1 / (1 + np.e), 1 - np.e / (1 + np.e), 1 - np.e /
            (1 + np.e), 0
        ]) / 2
        np.testing.assert_almost_equal(grad, expected_grad, decimal=15)

    def test_grad_loss_consistency(self):
        """Test longitudinal multinomial model gradient properties."""
        n_lags = np.repeat(9, 3).astype(dtype="uint64")
        sim = SimuSCCS(500, 36, 3, n_lags, None, "single_exposure", seed=42,
                       verbose=False)
        _, X, y, censoring, coeffs = sim.simulate()
        coeffs = np.hstack(coeffs)
        X, _, _ = LongitudinalFeaturesLagger(n_lags=n_lags) \
            .fit_transform(X, censoring)
        model = ModelSCCS(n_intervals=36, n_lags=n_lags)\
            .fit(X, y, censoring)
        self._test_grad(model, coeffs)
        X_sparse = [csr_matrix(x) for x in X]
        model = ModelSCCS(n_intervals=36, n_lags=n_lags)\
            .fit(X_sparse, y, censoring)
        self._test_grad(model, coeffs)

    def test_lipschitz_constant(self):
        """Test longitudinal multinomial model Lipschitz constant."""
        X = [
            np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype="float64"),
            np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype="float64")
        ]
        y = [
            np.array([0, 1, 0], dtype="int32"),
            np.array([0, 1, 0], dtype="int32")
        ]
        n_lags = np.repeat(1, 3).astype(dtype="uint64")
        X, _, _ = LongitudinalFeaturesLagger(n_lags=n_lags) \
            .fit_transform(X)
        model = ModelSCCS(n_intervals=3, n_lags=n_lags).fit(X, y)
        lip_constant = model.get_lip_max()
        expected_lip_constant = .5
        self.assertEqual(lip_constant, expected_lip_constant)

    def test_convergence_with_lags(self):
        """Test longitudinal multinomial model convergence."""
        n_intervals = 10
        n_samples = 800
        n_features = 2
        n_lags = np.repeat(2, n_features).astype(dtype="uint64")
        sim = SimuSCCS(n_samples, n_intervals, n_features, n_lags, None,
                       "multiple_exposures", seed=42)
        _, X, y, censoring, coeffs = sim.simulate()
        coeffs = np.hstack(coeffs)
        X, _, _ = LongitudinalFeaturesLagger(n_lags=n_lags) \
            .fit_transform(X, censoring)
        model = ModelSCCS(n_intervals=n_intervals, n_lags=n_lags).fit(
            X, y, censoring)
        solver = SVRG(max_iter=15, verbose=False)
        solver.set_model(model).set_prox(ProxZero())
        coeffs_svrg = solver.solve(step=1 / model.get_lip_max())
        np.testing.assert_almost_equal(coeffs, coeffs_svrg, decimal=1)

    def test_convergence_without_lags(self):
        """Test longitudinal multinomial model convergence."""
        n_intervals = 10
        n_samples = 600
        n_features = 2
        n_lags = np.repeat(0, n_features).astype(dtype="uint64")
        sim = SimuSCCS(n_samples, n_intervals, n_features, n_lags, None,
                       "multiple_exposures", seed=42, verbose=False)
        _, X, y, censoring, coeffs = sim.simulate()
        coeffs = np.hstack(coeffs)
        X, _, _ = LongitudinalFeaturesLagger(n_lags=n_lags) \
            .fit_transform(X, censoring)
        model = ModelSCCS(n_intervals=n_intervals, n_lags=n_lags).fit(
            X, y, censoring)
        solver = SVRG(max_iter=15, verbose=False)
        solver.set_model(model).set_prox(ProxZero())
        coeffs_svrg = solver.solve(step=1 / model.get_lip_max())
        np.testing.assert_almost_equal(coeffs, coeffs_svrg, decimal=1)

    def _test_grad(self, model, coeffs, delta_check_grad=1e-5,
                   delta_model_grad=1e-4):
        """Test that gradient is consistent with loss and that minimum is
            achievable with a small gradient
            """
        self.assertAlmostEqual(
            check_grad(model.loss, model.grad, coeffs), 0.,
            delta=delta_check_grad)
        # Check that minimum iss achievable with a small gradient
        coeffs_min = fmin_bfgs(model.loss, coeffs, fprime=model.grad,
                               disp=False)
        self.assertAlmostEqual(
            norm(model.grad(coeffs_min)), .0, delta=delta_model_grad)

    def test_sccs_serialize_and_compare(self):
        """Test serialization (cereal/pickle) of SCCS."""
        X = [
            np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype="float64"),
            np.array([[0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype="float64")
        ]
        y = [
            np.array([0, 1, 0], dtype="int32"),
            np.array([0, 1, 0], dtype="int32")
        ]
        n_lags = np.repeat(1, 3).astype(dtype="uint64")
        X, _, _ = LongitudinalFeaturesLagger(n_lags=n_lags) \
            .fit_transform(X)
        model = ModelSCCS(n_intervals=3, n_lags=n_lags).fit(X, y)

        pickled = pickle.loads(pickle.dumps(model))

        self.assertTrue(model._model.compare(pickled._model))

if __name__ == '__main__':
    unittest.main()
