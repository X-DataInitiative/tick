import unittest
import os
import numpy as np
from numpy.random import random, randint

from tick.inference import HawkesConditionalLaw


class Test(unittest.TestCase):
    def setUp(self):
        self.dim = 2
        np.random.seed(320982)
        self.timestamps = [np.cumsum(random(randint(20, 25))) * 10
                           for _ in range(self.dim)]
        self.model = HawkesConditionalLaw(n_quad=5)
        self.model.fit(self.timestamps)

    def test_hawkes_conditional_law_norm(self):
        """...Test HawkesConditionalLaw kernels norm estimation
        """
        np.testing.assert_array_almost_equal(self.model.kernels_norms,
                                             [[-0.7813225, -1.12426198],
                                              [-1.19782933, -1.75748751]])

    def test_hawkes_conditional_law_kernels(self):
        """...Test HawkesConditionalLaw kernel estimation
        """
        saved_phi_path = os.path.join(os.path.dirname(__file__),
                                      'hawkes_conditional_law_test-kernels.npy')
        saved_phi = np.load(saved_phi_path)
        np.testing.assert_array_almost_equal(self.model.kernels, saved_phi)

    def test_hawkes_conditional_law_baseline(self):
        """...Test HawkesConditionalLaw basleine estimation
        """
        np.testing.assert_array_almost_equal(self.model.baseline,
                                             [0.616481, 0.831181])

    def test_hawkes_conditional_mean_intensity(self):
        """...Test HawkesConditionalLaw mean intensity estimation
        """
        np.testing.assert_array_almost_equal(self.model.mean_intensity,
                                             [0.21470093, 0.20816257])

    def test_hawkes_quad_method(self):
        """...Test HawkesConditionalLaw estimates with different quadrature
        methods
        """
        model = HawkesConditionalLaw(n_quad=5, quad_method='gauss')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[-0.7813225, -1.12426198],
                                              [-1.19782933, -1.75748751]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='gauss-')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[-224.0366795, 36.68554746],
                                              [-174.78323768, 33.31661676]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='lin')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[15.09698511, 1.94961027],
                                              [-30.99208312, 14.75358309]])

        model = HawkesConditionalLaw(n_quad=5, quad_method='log')
        model.fit(self.timestamps)
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[-1.5177032, -4.56946934],
                                              [2.09935106, 3.36845494]])

    def test_hawkes_claw_method(self):
        """...Test HawkesConditionalLaw estimates with different conditional
        law methods
        """
        model = HawkesConditionalLaw(n_quad=5, claw_method='lin')
        model.incremental_fit(self.timestamps, compute=False)
        model.compute()
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[-0.7813225, -1.12426198],
                                              [-1.19782933, -1.75748751]])

        model = HawkesConditionalLaw(n_quad=5, claw_method='log')
        model.incremental_fit(self.timestamps)
        np.testing.assert_array_almost_equal(model.kernels_norms,
                                             [[0.42528942, -0.49200346],
                                              [-1.18794187, -6.19311372]])

if __name__ == "__main__":
    unittest.main()
