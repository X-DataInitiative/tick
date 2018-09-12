# License: BSD 3 clause

import unittest

import numpy as np

from tick.hawkes.inference import HawkesSumGaussians


class Test(unittest.TestCase):
    def setUp(self):
        self.int_1 = 4
        self.int_2 = 6
        self.float_1 = 0.3
        self.float_2 = 0.2

    def test_hawkes_sumgaussians_solution(self):
        """...Test solution obtained by HawkesSumGaussians on toy timestamps
        """
        events = [[
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ], [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19])
        ]]

        n_nodes = len(events[0])
        n_gaussians = 3
        max_mean_gaussian = 5
        step_size = 1e-3
        C = 10
        lasso_grouplasso_ratio = 0.7

        baseline_start = np.zeros(n_nodes) + .2
        amplitudes_start = np.zeros((n_nodes, n_nodes, n_gaussians)) + .2

        learner = HawkesSumGaussians(
            n_gaussians=n_gaussians, max_mean_gaussian=max_mean_gaussian,
            step_size=step_size, C=C,
            lasso_grouplasso_ratio=lasso_grouplasso_ratio, n_threads=3,
            max_iter=11, verbose=False, em_max_iter=3)
        learner.fit(events[0], baseline_start=baseline_start,
                    amplitudes_start=amplitudes_start)

        baseline = np.array([0.0979586, 0.15552228])

        amplitudes = np.array([[[0.20708954, -0.00627318, 0.08388442],
                                [-0.00341803, 0.34805652, -0.00687372]],
                               [[-0.00341635, 0.1608013, 0.05531324],
                                [-0.00342652, -0.00685425, 0.19046195]]])

        np.testing.assert_array_almost_equal(learner.baseline, baseline,
                                             decimal=6)
        np.testing.assert_array_almost_equal(learner.amplitudes, amplitudes,
                                             decimal=6)

        kernel_values = np.array([
            -0.00068796, 0.01661161, 0.08872543, 0.21473618, 0.25597692,
            0.15068586, 0.04194497, 0.00169372, -0.00427233, -0.00233042
        ])
        kernels_norm = np.array([[0.28470077, 0.33776477],
                                 [0.21269818, 0.18018118]])

        np.testing.assert_almost_equal(
            learner.get_kernel_values(0, 1, np.linspace(0, 4, 10)),
            kernel_values)
        np.testing.assert_almost_equal(learner.get_kernel_norms(),
                                       kernels_norm)

        means_gaussians = np.array([0., 1.66666667, 3.33333333])
        std_gaussian = 0.5305164769729844
        np.testing.assert_array_almost_equal(learner.means_gaussians,
                                             means_gaussians)
        self.assertEqual(learner.std_gaussian, std_gaussian)

        learner.n_gaussians = learner.n_gaussians + 1
        means_gaussians = np.array([0., 1.25, 2.5, 3.75])
        std_gaussian = 0.3978873577297384
        np.testing.assert_array_almost_equal(learner.means_gaussians,
                                             means_gaussians)
        self.assertEqual(learner.std_gaussian, std_gaussian)

    def test_hawkes_sumgaussians_set_data(self):
        """...Test set_data method of Hawkes SumGaussians
        """
        events = [[
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ], [
            np.array([2, 3.2, 11.4, 12.8, 45]),
            np.array([2, 3, 8.8, 9, 15.3, 19])
        ]]

        learner = HawkesSumGaussians(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        events = [
            np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
            np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
        ]

        learner = HawkesSumGaussians(1)
        learner._set_data(events)
        self.assertEqual(learner.n_nodes, 2)

        msg = "All realizations should have 2 nodes, but realization 1 has " \
              "1 nodes"
        with self.assertRaisesRegex(RuntimeError, msg):
            events = [[
                np.array([1, 1.2, 3.4, 5.8, 10.3, 11, 13.4]),
                np.array([2, 5, 8.3, 9.10, 15, 18, 20, 33])
            ], [np.array([2, 3.2, 11.4, 12.8, 45])]]
            learner._set_data(events)

    def test_hawkes_sumgaussians_parameters(self):
        """...Test that hawkes sumgaussians parameters are correctly linked
        """
        learner = HawkesSumGaussians(1, n_gaussians=self.int_1)
        self.assertEqual(learner.n_gaussians, self.int_1)
        self.assertEqual(learner._learner.get_n_gaussians(), self.int_1)
        learner.n_gaussians = self.int_2
        self.assertEqual(learner.n_gaussians, self.int_2)
        self.assertEqual(learner._learner.get_n_gaussians(), self.int_2)

        learner = HawkesSumGaussians(max_mean_gaussian=self.float_1)
        self.assertEqual(learner.max_mean_gaussian, self.float_1)
        self.assertEqual(learner._learner.get_max_mean_gaussian(),
                         self.float_1)
        learner.max_mean_gaussian = self.float_2
        self.assertEqual(learner.max_mean_gaussian, self.float_2)
        self.assertEqual(learner._learner.get_max_mean_gaussian(),
                         self.float_2)

        learner = HawkesSumGaussians(1, step_size=self.float_1)
        self.assertEqual(learner.step_size, self.float_1)
        self.assertEqual(learner._learner.get_step_size(), self.float_1)
        learner.step_size = self.float_2
        self.assertEqual(learner.step_size, self.float_2)
        self.assertEqual(learner._learner.get_step_size(), self.float_2)

    def test_hawkes_sumgaussians_lasso_grouplasso_ratio_parameter(self):
        """...Test that hawkes sumgaussians lasso_grouplasso_ratio parameter is 
        correctly linked
        """
        # First learner initialization
        C = 5e-3
        learner = HawkesSumGaussians(1, lasso_grouplasso_ratio=self.float_1,
                                     C=C)
        strength_lasso = self.float_1 / learner.C
        strength_grouplasso = (1. - self.float_1) / learner.C
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(),
                         strength_grouplasso)
        self.assertEqual(learner.C, C)

        # change lasso_grouplasso_ratio
        learner.lasso_grouplasso_ratio = self.float_2
        strength_lasso = self.float_2 / learner.C
        strength_grouplasso = (1. - self.float_2) / learner.C
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(),
                         strength_grouplasso)
        self.assertEqual(learner.lasso_grouplasso_ratio, self.float_2)
        self.assertEqual(learner.C, C)

    def test_hawkes_sumgaussians_C_parameter(self):
        """...Test that hawkes sumgaussians C parameter is correctly linked
        """
        # First leaner initialization
        lasso_grouplasso_ratio = 0.3
        learner = HawkesSumGaussians(
            1, C=self.float_1, lasso_grouplasso_ratio=lasso_grouplasso_ratio)
        strength_lasso = learner.lasso_grouplasso_ratio / self.float_1
        strength_grouplasso = (1. - learner.lasso_grouplasso_ratio) / \
                              self.float_1
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(),
                         strength_grouplasso)
        self.assertEqual(learner.lasso_grouplasso_ratio,
                         lasso_grouplasso_ratio)

        # Change C
        learner.C = self.float_2
        strength_lasso = learner.lasso_grouplasso_ratio / self.float_2
        strength_grouplasso = (1. - learner.lasso_grouplasso_ratio) / \
                              self.float_2
        self.assertEqual(learner.strength_lasso, strength_lasso)
        self.assertEqual(learner.strength_grouplasso, strength_grouplasso)
        self.assertEqual(learner._learner.get_strength_lasso(), strength_lasso)
        self.assertEqual(learner._learner.get_strength_grouplasso(),
                         strength_grouplasso)
        self.assertAlmostEqual(learner.C, self.float_2)
        self.assertEqual(learner.lasso_grouplasso_ratio,
                         lasso_grouplasso_ratio)


if __name__ == '__main__':
    unittest.main()
