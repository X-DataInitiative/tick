import numpy as np

from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti
from tick.inference import HawkesDual, HawkesExpKern
from tick.plot import plot_history, plot_hawkes_kernel_norms

simulate = False

def estimation_error(estimated, original):
    return np.linalg.norm(original - estimated) ** 2 / \
           np.linalg.norm(original) ** 2

end_time = 10000
n_realizations = 5
decay = 3.

n_nodes = 100
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
    np.save('timestamps_simulated.npy', timestamps)

else:
    timestamps = np.load('timestamps_simulated.npy')
    timestamps = [[t for t in timestamp] for timestamp in timestamps]

l_l2sq = 5e-1


hawkes_svrg_learners = []
steps = np.logspace(-4, -7, 0)
for step in steps:
    hawkes_learner = HawkesExpKern(decay, gofit='likelihood', verbose=True,
                                   step=step, C=1/l_l2sq, penalty='l2', solver='svrg',
                                   record_every=1)
    hawkes_learner.fit(timestamps, start=0.1)
    hawkes_svrg_learners += [hawkes_learner]


hawkes_lbgfsb = HawkesExpKern(decay, gofit='likelihood', verbose=True,
                              C=1/l_l2sq, penalty='l2', solver='l-bfgs-b',
                              record_every=1)
hawkes_lbgfsb._model_obj.n_threads = 4
hawkes_lbgfsb.fit(timestamps)

print(hawkes_lbgfsb._model_obj.n_jumps)

hawkes_dual = HawkesDual(decay, l_l2sq, verbose=True, record_every=1,
                         n_threads=4)
hawkes_dual.fit(timestamps)


print('ESTIMATION ERROR',
      estimation_error(hawkes_dual.adjacency, hawkes_exp_kernels.adjacency))

all_learners = [hawkes_dual] + hawkes_svrg_learners + [hawkes_lbgfsb]
labels = ["dual"] + ['SVRG {:.2g}'.format(step) for step in steps] + \
         ['L-BFGS-B']

plot_history(all_learners, dist_min=True, log_scale=True,
             labels=labels)

# plot_hawkes_kernel_norms(hawkes_dual)