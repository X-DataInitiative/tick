# License: BSD 3 clause

import unittest

import numpy as np
from scipy.stats import norm as normal

from tick.prox import ProxSlope
from tick.prox.tests.prox import TestProx


class ProxSlopeTest(object):
    def get_coeffs_weak(self, n_coeffs):
        return (-1) ** np.arange(n_coeffs) * \
            np.sqrt(2 * np.log(np.linspace(1, 10, n_coeffs) * n_coeffs))

    def get_coeffs_strong(self, n_coeffs):
        return 5 * self.get_coeffs_weak(n_coeffs)

    def get_weights_bh(self, strength, fdr, size):
        tmp = fdr / (2 * size)
        return strength * normal.ppf(1 - tmp * np.arange(1, size + 1))

    def test_ProxSlope_call(self):
        """...Test_of ProxSlope.call
        """
        n_coeffs = 30
        np.random.seed(seed=123)
        coeffs = np.zeros(n_coeffs).astype(self.dtype)
        coeffs[:10] = self.get_coeffs_strong(10).astype(self.dtype)
        y = coeffs + 3 * np.random.randn(n_coeffs).astype(self.dtype)
        fdr = 0.6
        strength = 4.
        step = 1.
        prox = ProxSlope(strength=strength, fdr=fdr).astype(self.dtype)

        x_sl1_truth \
            = np.array([2.13923654, -3.53844034, 7.31022147, -9.87610726,
                        6.03085298, -3.53844034, 2.13923654, -9.08606547,
                        9.87610726, -9.87610726, -0., -0., 0.08661054, -0.,
                        -0., -0., 1.78604443, 1.78604443, 0., 0., 0.,
                        0.08661054, -0., 0., -0., -0., 0., -0.08661054, -0.,
                        -0.]).astype(self.dtype)
        np.testing.assert_almost_equal(
            prox.call(y, step=step), x_sl1_truth, decimal=self.decimal_places)

        strength = 2.
        step = 2.
        prox.strength = strength
        np.testing.assert_almost_equal(
            prox.call(y, step=step), x_sl1_truth, decimal=self.decimal_places)

        prox.range = (10, 20)
        strength = 4.
        step = 1.
        prox.strength = strength

        x_sl1_truth \
            = np.array([7.47293832, -9.24669781, 13.88963598, -18.0998993,
                        12.24994736, -9.35363322, 7.29476114,  -16.08880976,
                        18.79749156, -17.7744925, -0., -0., 0., -0., -0., -0.,
                        0., 0., 0., 0., 2.21210573, 4.47219608, -2.80750161,
                        3.52748713, -3.761642, -1.91325451, 2.72131559,
                        -4.2860421, -0.42020616, -2.58526469]).astype(self.dtype)
        np.testing.assert_almost_equal(
            prox.call(y, step=step), x_sl1_truth, decimal=self.decimal_places)

        strength = 2.
        step = 2.
        prox.strength = strength
        np.testing.assert_almost_equal(
            prox.call(y, step=step), x_sl1_truth, decimal=self.decimal_places)

    def test_ProxSlope_weights_bh(self):
        """...Test of ProxSlope weights computation
        """
        n_samples = 5000
        n_outliers = int(0.1 * n_samples)
        interc0 = np.zeros(n_samples).astype(self.dtype)
        interc0[:n_outliers] = self.get_coeffs_strong(n_outliers).astype(
            self.dtype)
        y = interc0 + 3 * np.random.randn(interc0.shape[0]).astype(self.dtype)

        strength = 2.5
        prox = ProxSlope(strength=strength).astype(self.dtype)
        prox.value(y)
        prox.strength = 20
        size = y.shape[0]
        tmp = prox.fdr / (2 * size)
        weights = strength * normal.ppf(1 - tmp * np.arange(1, size + 1))

        prox_weights = np.array(
            [prox._prox.get_weight_i(i) for i in range(size)])
        np.testing.assert_almost_equal(prox_weights, weights,
                                       decimal=self.decimal_places)

        strength = 2.5
        prox = ProxSlope(strength=strength, range=(300,
                                                   3000)).astype(self.dtype)
        prox.value(y)
        prox.strength = 20
        size = prox.range[1] - prox.range[0]
        tmp = prox.fdr / (2 * size)
        weights = strength * normal.ppf(1 - tmp * np.arange(1, size + 1))

        prox_weights = np.array(
            [prox._prox.get_weight_i(i) for i in range(size)])
        np.testing.assert_almost_equal(prox_weights, weights,
                                       decimal=self.decimal_places)

    def test_ProxSlope_value(self):
        """...Test of ProxSlope.value
        """
        np.random.seed(1337)
        n_samples = 5000
        n_outliers = int(0.1 * n_samples)
        interc0 = np.zeros(n_samples)
        interc0[:n_outliers] = self.get_coeffs_strong(n_outliers)
        y = (interc0 + 3 * np.random.randn(interc0.shape[0])).astype(
            self.dtype)

        fdr = 0.6
        strength = 2.5
        prox = ProxSlope(strength=strength, fdr=fdr).astype(self.dtype)
        prox_value = prox.value(y)

        weights = self.get_weights_bh(strength, fdr, size=y.shape[0])
        y_abs = np.abs(y)
        sub_y = y_abs
        value = sub_y[np.argsort(-sub_y)].dot(weights)
        places = 4

        self.assertAlmostEqual(prox_value, value, delta=value * 1e-7)

        strength = 7.4
        fdr = 0.2
        prox.strength = strength
        prox.fdr = fdr
        prox_value = prox.value(y)

        weights = self.get_weights_bh(strength, fdr, size=y.shape[0])
        y_abs = np.abs(y)
        sub_y = y_abs
        value = sub_y[np.argsort(-sub_y)].dot(weights)
        self.assertAlmostEqual(prox_value, value, delta=value * 1e-7)

        prox.range = (300, 3000)
        fdr = 0.6
        strength = 2.5
        prox.strength = strength
        prox.fdr = fdr
        prox_value = prox.value(y)

        a, b = prox.range
        size = b - a
        weights = self.get_weights_bh(strength, fdr, size=size)
        y_abs = np.abs(y[a:b])
        sub_y = y_abs
        value = sub_y[np.argsort(-sub_y)].dot(weights)
        self.assertAlmostEqual(prox_value, value, delta=value * 1e-7)

        prox.range = (300, 3000)
        strength = 7.4
        fdr = 0.2
        prox.strength = strength
        prox.fdr = fdr
        prox_value = prox.value(y)

        a, b = prox.range
        size = b - a
        weights = self.get_weights_bh(strength, fdr, size=size)
        y_abs = np.abs(y[a:b])
        sub_y = y_abs
        value = sub_y[np.argsort(-sub_y)].dot(weights)
        self.assertAlmostEqual(prox_value, value, delta=value * 1e-7)


class ProxSlopeTestFloat32(TestProx, ProxSlopeTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float32", **kwargs)


class ProxSlopeTestFloat64(TestProx, ProxSlopeTest):
    def __init__(self, *args, **kwargs):
        TestProx.__init__(self, *args, dtype="float64", **kwargs)


if __name__ == '__main__':
    unittest.main()
