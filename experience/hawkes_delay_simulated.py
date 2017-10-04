import numpy as np
import matplotlib.pyplot as plt

from tick.base import TimeFunction
from tick.simulation import SimuHawkesSumExpKernels, SimuHawkesMulti, \
    HawkesKernelTimeFunc, SimuHawkes
from tick.inference import HawkesDual, HawkesSumExpKern
from tick.plot import plot_history, plot_hawkes_kernel_norms, \
    plot_hawkes_kernels

simulate = False

def estimation_error(estimated, original):
    return np.linalg.norm(original - estimated) ** 2 / \
           np.linalg.norm(original) ** 2


n_realizations = 5

end_time = 10000
t_values = np.array([0., 2.5, 8.5, 10.], dtype=float)
y_values = np.array([0., .05, 0., 0.], dtype=float)
tf = TimeFunction([t_values, y_values],
                   inter_mode=TimeFunction.InterConstRight, dt=0.1)
delay_kernel = HawkesKernelTimeFunc(tf)
hawkes_delay = SimuHawkes(baseline=[0.2], kernels=[[delay_kernel]],
                          end_time=end_time, verbose=False, seed=1039)

if simulate:
    multi = SimuHawkesMulti(hawkes_delay, n_simulations=n_realizations,
                            n_threads=4)

    multi.end_time = [(i + 1) / n_realizations * end_time
                      for i in range(n_realizations)]
    multi.simulate()

    timestamps = multi.timestamps
    np.save('timestamps_delay_simulated.npy', timestamps)

else:
    timestamps = np.load('timestamps_delay_simulated.npy')
    timestamps = [[t for t in timestamp] for timestamp in timestamps]


decays = np.logspace(-1, .5, 6)
l_l2sq = 5e-1


hawkes_svrg_learners = []
steps = np.logspace(-6, -7, 0)
for step in steps:
    hawkes_learner = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                   step=step, C=1/l_l2sq, penalty='l2', solver='svrg',
                                   record_every=1, max_iter=100)
    hawkes_learner.fit(timestamps, start=0.1)
    hawkes_svrg_learners += [hawkes_learner]


hawkes_lbgfsb = HawkesSumExpKern(decays, gofit='likelihood', verbose=True,
                                 C=1 / l_l2sq, penalty='l2', solver='l-bfgs-b',
                                 record_every=1, max_iter=100, tol=0)
# hawkes_lbgfsb._prox_obj.positive = False
hawkes_lbgfsb._model_obj.n_threads = 4
hawkes_lbgfsb.fit(timestamps)

print(hawkes_lbgfsb._model_obj.n_jumps)

hawkes_dual = HawkesDual(decays, l_l2sq, verbose=True, record_every=1,
                         n_threads=4, max_iter=50, tol=1e-13)

hawkes_dual.fit(timestamps)


# print(hawkes_exp_kernels.adjacency[0, 0])
# print(hawkes_dual.adjacency[0, 0])
# print(hawkes_lbgfsb.adjacency[0, 0])


all_learners = [hawkes_dual] + hawkes_svrg_learners + [hawkes_lbgfsb]
labels = ["dual"] + ['SVRG {:.2g}'.format(step) for step in steps] + \
         ['L-BFGS-B']


# plot_hawkes_kernel_norms(hawkes_dual)

# plot_hawkes_kernels(hawkes_dual, hawkes=hawkes_delay)
#
# plot_hawkes_kernels(hawkes_lbgfsb, hawkes=hawkes_delay)
print(hawkes_lbgfsb.adjacency)
print(hawkes_dual.adjacency)

# print(hawkes_lbgfsb._solver_obj.objective(hawkes_lbgfsb.coeffs), hawkes_dual.objective(hawkes_lbgfsb.coeffs))

dual_coeffs = hawkes_dual.coeffs
loss_bfgs_dual_coeffs = hawkes_lbgfsb._solver_obj.model.loss(dual_coeffs)
loss_dual_dual_coeffs = hawkes_dual._learner.loss(dual_coeffs)

print(loss_bfgs_dual_coeffs, loss_dual_dual_coeffs)
print(loss_bfgs_dual_coeffs, loss_dual_dual_coeffs)
print(hawkes_lbgfsb._solver_obj.objective(dual_coeffs), hawkes_dual.objective(dual_coeffs))

# fig = plot_history(all_learners, dist_min=True, log_scale=True,
#                    labels=labels, show=False)
#
# fig.gca().set_ylim([None, 1])
#
# plt.show()

plot_hawkes_kernels(hawkes_dual, hawkes=hawkes_delay, show=False)
plt.savefig("hawkes_dual_delay_kernel.png")
plot_hawkes_kernels(hawkes_lbgfsb, hawkes=hawkes_delay, show=False)
plt.savefig("hawkes_bfgs_delay_kernel.png")