import numpy as np

from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesDual, HawkesExpKern
from tick.plot import plot_history, plot_hawkes_kernel_norms

np.random.seed(23928)

def estimation_error(estimated, original):
    return np.linalg.norm(original - estimated) ** 2 / \
           np.linalg.norm(original) ** 2

simulate = False

end_time = 10000
n_realizations = 5
decay = 3.

n_nodes = 30
baseline = np.abs(np.random.normal(scale=1 / n_nodes, size=n_nodes))
adjacency = np.abs(np.random.normal(size=(n_nodes, n_nodes)))

hawkes_exp_kernels = SimuHawkesExpKernels(
    adjacency=adjacency, decays=decay, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

hawkes_exp_kernels.adjust_spectral_radius(0.8)

if simulate:
    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

    multi.end_time = [(i + 1) / n_realizations * end_time
                      for i in range(n_realizations)]
    multi.simulate()

    timestamps = multi.timestamps[-1]
    np.save('timestamps_speedup.npy', timestamps)

else:
    timestamps = np.load('timestamps_speedup.npy')
    timestamps = [timestamp for timestamp in timestamps]

l_l2sq = 0.1
hawkes_dual = HawkesDual(decay, l_l2sq, verbose=True, record_every=5,
                         n_threads=4)
hawkes_dual.fit(timestamps)
#
# plot_history([hawkes_dual], dist_min=True, log_scale=True)
