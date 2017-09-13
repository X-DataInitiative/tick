import numpy as np

from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesDual, HawkesExpKern
import matplotlib.pyplot as plt

np.random.seed(23928)

def estimation_error(estimated, original):
    return np.linalg.norm(original - estimated) ** 2 / \
           np.linalg.norm(original) ** 2

simulate = False

end_time = 20000
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

    timestamps = multi.timestamps
    np.save('timestamps_speedup.npy', timestamps)

else:
    timestamps = np.load('timestamps_speedup.npy')
    timestamps = [[t for t in timestamp] for timestamp in timestamps]

l_l2sq = 0.1

n_threads = range(1, 10)
hawkes_duals = []

for n_thread in n_threads:
    hawkes_dual = HawkesDual(decay, l_l2sq, verbose=True, record_every=5,
                             n_threads=n_thread)
    hawkes_dual.fit(timestamps)
    hawkes_duals += [hawkes_dual]

time_one = hawkes_duals[0].history.last_values['time']
plt.plot([n_threads[0], n_threads[-1]], [1, n_threads[-1]], ls='--',
         lw=1, c='black')

y = [time_one / hawkes_dual.history.last_values['time'] for hawkes_dual in hawkes_duals]
plt.plot(n_threads, y)
plt.xlabel('number of cores')
plt.ylabel('speedup')

plt.show()
