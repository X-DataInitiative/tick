# License: BSD 3 clause

import unittest

import numpy as np
import itertools

from tick.hawkes import HawkesSumExpKern
from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(238924)
        decays = np.array([1., 3.])
        self.n_nodes = 2

        baseline = np.ones(self.n_nodes)
        adjacency = np.random.rand(self.n_nodes, self.n_nodes, len(decays))
        self.hawkes_simu = SimuHawkesSumExpKernels(
            decays=decays, baseline=baseline, adjacency=adjacency,
            max_jumps=100, verbose=False, seed=32098)
        self.hawkes_simu.simulate()

    def test_plot_hawkes_kernels(self):
        """...Test plot_hawkes_history rendering given a fitted Hawkes
        learner
        """
        decays = np.array([1.5, 3.5])
        hawkes_sumexp = HawkesSumExpKern(decays, max_iter=0)
        # We set some specific coeffs to be free from any future learner
        # modifications
        # With 0 iteration and coeffs as start point it should remain there
        coeffs = np.array(
            [0.99, 0.99, 0.55, 0.37, 0.39, 0.16, 0.63, 0.49, 0.49, 0.30])
        hawkes_sumexp.fit(self.hawkes_simu.timestamps, start=coeffs)

        n_points = 10
        for support in [None, 4]:
            fig = plot_hawkes_kernels(hawkes_sumexp, hawkes=self.hawkes_simu,
                                      show=False, n_points=n_points,
                                      support=support)

            if support is None:
                max_support = hawkes_sumexp.get_kernel_supports().max() * 1.2
            else:
                max_support = support

            for i, j in itertools.product(range(self.n_nodes), repeat=2):
                index = i * self.n_nodes + j
                ax = fig.axes[index]
                ax_t_axis, ax_estimated_kernel = ax.lines[0].get_xydata().T
                t_axis = np.linspace(0, max_support, n_points)
                np.testing.assert_array_equal(ax_t_axis, t_axis)

                estimated_kernel = hawkes_sumexp.get_kernel_values(
                    i, j, t_axis)
                np.testing.assert_array_equal(ax_estimated_kernel,
                                              estimated_kernel)

                _, ax_true_kernel = ax.lines[1].get_xydata().T
                true_kernel = self.hawkes_simu.kernels[i, j].get_values(t_axis)
                np.testing.assert_array_equal(ax_true_kernel, true_kernel)


if __name__ == '__main__':
    unittest.main()
