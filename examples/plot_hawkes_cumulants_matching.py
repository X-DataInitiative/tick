"""
=======================================
Fit Hawkes kernel norms using cumulants
=======================================

This non parametric Hawkes cumulants matching
(`tick.hawkes.HawkesCumulantMatching`) algorithm estimates directly
kernels norms without making any assumption on kernel shapes.

It has been originally described in this paper:

Achab, M., Bacry, E., Gaiffas, S., Mastromatteo, I., & Muzy, J. F.
(2017, July). Uncovering causality from multivariate Hawkes integrated
cumulants.
`In International Conference on Machine Learning (pp. 1-10)`_.

.. _In International Conference on Machine Learning (pp. 1-10): http://proceedings.mlr.press/v70/achab17a.html
"""

import numpy as np

from tick.hawkes import (HawkesCumulantMatching, SimuHawkesExpKernels,
                         SimuHawkesMulti)
from tick.plot import plot_hawkes_kernel_norms

np.random.seed(7168)

n_nodes = 3
baselines = 0.3 * np.ones(n_nodes)
decays = 0.5 + np.random.rand(n_nodes, n_nodes)
adjacency = np.array([
    [1, 1, -0.5],
    [0, 1, 0],
    [0, 0, 2],
], dtype=float)

adjacency /= 4

end_time = 1e5
integration_support = 5
n_realizations = 5

simu_hawkes = SimuHawkesExpKernels(baseline=baselines, adjacency=adjacency,
                                   decays=decays, end_time=end_time,
                                   verbose=False, seed=7168)
simu_hawkes.threshold_negative_intensity(True)

multi = SimuHawkesMulti(simu_hawkes, n_simulations=n_realizations,
                        n_threads=-1)
multi.simulate()

nphc = HawkesCumulantMatching(integration_support, cs_ratio=.15, tol=1e-10,
                              step=0.3)

nphc.fit(multi.timestamps)
plot_hawkes_kernel_norms(nphc)
