import numpy as np

from tick.simulation import SimuHawkesSumExpKernels, SimuHawkesMulti
from tick.inference import HawkesDual, HawkesSumExpKern
from tick.plot import plot_history, plot_hawkes_kernel_norms, \
    plot_hawkes_kernels

simulate = True

def estimation_error(estimated, original):
    return np.linalg.norm(original - estimated) ** 2 / \
           np.linalg.norm(original) ** 2


end_time = 10000
n_realizations = 5

n_nodes = 10
decays = [1., 5., 12.]
n_decays = len(decays)
baseline = np.abs(np.random.normal(scale=1 / n_nodes, size=n_nodes))
adjacency = np.abs(np.random.normal(size=(n_nodes, n_nodes, n_decays)))
baseline[0] = 0.5
adjacency[0, 0, 0] = 3
adjacency[0, 0, 1] = -3
adjacency[0, 0, 0] = 3

hawkes_exp_kernels = SimuHawkesSumExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline,
    end_time=end_time, verbose=False, seed=1039)

hawkes_exp_kernels.adjust_spectral_radius(0.7)

if simulate:
    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations,
                            n_threads=4)

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
steps = np.logspace(-6, -7, 0)
for step in steps:
    hawkes_learner = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                   step=step, C=1/l_l2sq, penalty='l2', solver='svrg',
                                   record_every=1, max_iter=30)
    hawkes_learner.fit(timestamps, start=0.1)
    hawkes_svrg_learners += [hawkes_learner]


hawkes_lbgfsb = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                 C=1 / l_l2sq, penalty='l2', solver='l-bfgs-b',
                                 record_every=1, max_iter=30)
hawkes_lbgfsb._model_obj.n_threads = 4
hawkes_lbgfsb.fit(timestamps)

print(hawkes_lbgfsb._model_obj.n_jumps)

hawkes_dual = HawkesDual(decays, l_l2sq, verbose=True, record_every=1,
                         n_threads=4, max_iter=30, tol=1e-13)

hawkes_dual.fit(timestamps)


print('ESTIMATION ERROR',
      estimation_error(hawkes_dual.adjacency, hawkes_exp_kernels.adjacency))

print(hawkes_exp_kernels.adjacency[0, 0])
print(hawkes_dual.adjacency[0, 0])
print(hawkes_lbgfsb.adjacency[0, 0])


all_learners = [hawkes_dual] + hawkes_svrg_learners + [hawkes_lbgfsb]
labels = ["dual"] + ['SVRG {:.2g}'.format(step) for step in steps] + \
         ['L-BFGS-B']

plot_history(all_learners, dist_min=True, log_scale=True,
             labels=labels)

# plot_hawkes_kernel_norms(hawkes_dual)

# plot_hawkes_kernels(hawkes_dual, hawkes=hawkes_exp_kernels)
