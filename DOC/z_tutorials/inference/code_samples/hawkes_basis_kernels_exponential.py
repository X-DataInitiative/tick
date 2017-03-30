import numpy as np

from tick.inference import HawkesBasisKernels
from tick.plot import plot_hawkes_kernels
from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti

run_time = 5000

n_nodes = 2

n_realizations = 10

baseline = [0.1, 0.1]
adjacency = [[0.3, 0.6], [0.15, 0.21]]
decay = 2.

hawkes = SimuHawkesExpKernels(baseline=baseline, adjacency=adjacency,
                              decays=decay, end_time=run_time, seed=2309,
                              verbose=False)
multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations)
multi.simulate()
ticks = multi.timestamps

kernel_support = 5
kernel_size = 50
n_basis = 1
C = 0.1

baseline_start = np.zeros(n_nodes) + .2
amplitudes_start = np.zeros((n_nodes, n_nodes, n_basis)) + .4
basis_kernels_start = np.zeros((n_basis, kernel_size)) + 0.1

learner = HawkesBasisKernels(kernel_support, n_basis=n_basis,
                             kernel_size=kernel_size, C=C,
                             n_threads=1, max_iter=30,
                             ode_max_iter=1000,
                             tol=1e-2, ode_tol=1e-3)
learner.fit(ticks, baseline_start=baseline_start,
            amplitudes_start=amplitudes_start,
            basis_kernels_start=basis_kernels_start)

plot_hawkes_kernels(learner, hawkes=hawkes)
