# License: BSD 3 clause

import os
import unittest

import numpy as np
from numpy.random import random, randint

from tick.base.inference import InferenceTest
from tick.hawkes.inference import HawkesConditionalLaw


class Test(InferenceTest):
    def setUp(self):
        self.dim = 2
        np.random.seed(320982)
        self.timestamps = [
            np.cumsum(random(randint(20, 25))) * 10 for _ in range(self.dim)
        ]
        self.model = HawkesConditionalLaw(n_quad=5)
        self.model.fit(self.timestamps)

    def test_hawkes_conditional_law_norm(self):
        """...Test HawkesConditionalLaw kernels norm estimation
        """
        np.testing.assert_array_almost_equal(
            self.model.kernels_norms,
            [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]])

    def test_hawkes_conditional_law_kernels(self):
        """...Test HawkesConditionalLaw kernel estimation
        """
        saved_phi_path = os.path.join(
            os.path.dirname(__file__),
            'hawkes_conditional_law_test-kernels.npy')
        saved_phi = np.load(saved_phi_path)
        np.testing.assert_array_almost_equal(self.model.kernels, saved_phi)

    def test_hawkes_conditional_law_baseline(self):
        """...Test HawkesConditionalLaw baseline estimation
        """
        np.testing.assert_array_almost_equal(self.model.baseline,
                                             [0.61213243, 0.808886425])

    def test_hawkes_conditional_mean_intensity(self):
        """...Test HawkesConditionalLaw mean intensity estimation
        """
        np.testing.assert_array_almost_equal(self.model.mean_intensity,
                                             [0.208121177, 0.208121177])

    def test_hawkes_quad_method(self):
        """...Test HawkesConditionalLaw estimates with different quadrature
        methods
        """
        model = HawkesConditionalLaw(n_quad=5, quad_method='gauss')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='gauss-')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[-77.76904711, 0.69985519], [-42.87140913, 0.13607425]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='lin')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[7.92561315, 1.74540188], [-28.57048537, 10.77926367]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='log')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[35.70738975, 18.96902121], [-51.69638233, -30.33936597]])

    def test_hawkes_claw_method(self):
        """...Test HawkesConditionalLaw estimates with different conditional
        law methods
        """
        model = HawkesConditionalLaw(n_quad=5, claw_method='lin')
        model.incremental_fit(self.timestamps, compute=False)
        model.compute()
        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[-0.81130911, -1.12992177], [-1.16313257, -1.72348019]])

        model = HawkesConditionalLaw(n_quad=5, claw_method='log')
        model.incremental_fit(self.timestamps)

        np.testing.assert_array_almost_equal(
            model.kernels_norms,
            [[0.46108403, -0.09467477], [-0.04787463, -3.82917571]])

    def test_incremental_fit(self):
        # This should not raise a warning
        self.model.incremental_fit(self.timestamps, compute=False)

        msg = 'compute\(\) method was already called, computed ' \
              'kernels will be updated.'
        with self.assertWarnsRegex(UserWarning, msg):
            self.model.incremental_fit(self.timestamps, compute=True)

        new_model = HawkesConditionalLaw(n_quad=5, claw_method='lin')
        new_model.incremental_fit(self.timestamps, compute=False)

        # This should not raise a warning
        new_model.incremental_fit(self.timestamps, compute=True)

        with self.assertWarnsRegex(UserWarning, msg):
            new_model.incremental_fit(self.timestamps, compute=True)


if __name__ == "__main__":
    unittest.main()
