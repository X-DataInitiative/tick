import numpy as np

import matplotlib.pyplot as plt

from tick.simulation import SimuHawkesSumExpKernels, SimuHawkesMulti
from tick.inference import HawkesDual, HawkesSumExpKern, HawkesEM
from tick.plot import plot_history, plot_hawkes_kernel_norms


def estimation_error(estimated, original):
    sq = np.linalg.norm(original.ravel() - estimated.ravel()) ** 2 / \
         np.linalg.norm(original.ravel()) ** 2
    return np.sqrt(sq)


np.random.seed(30944)

end_time = 10000
n_realizations = 5

n_nodes = 10
decays = [1., 5., 12.]
n_decays = len(decays)
baseline = np.abs(np.random.normal(scale=1 / n_nodes, size=n_nodes))

fig, ax_list = plt.subplots(2, 2, figsize=(7, 5.5))

for n_plot, positive in enumerate([True, False]):
    if positive:
        adjacency = 4 + np.abs(
            np.random.normal(size=(n_nodes, n_nodes, n_decays)))
        max_iter = 30
    else:
        adjacency = 0.5 + np.random.normal(size=(n_nodes, n_nodes, n_decays))
        max_iter = 60

    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=1039)

    hawkes_exp_kernels.adjust_spectral_radius(0.7)
    hawkes_exp_kernels.threshold_negative_intensity()

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations,
                            n_threads=4)

    multi.end_time = [(i + 1) / n_realizations * end_time
                      for i in range(n_realizations)]
    multi.simulate()

    timestamps = multi.timestamps

    if positive:
        l_l2sq_list = np.logspace(-0.5, -1.5, 1)
    else:
        l_l2sq_list = np.logspace(-1.3, -1.7, 1)

    hawkes_bfgs_learners = []
    hawkes_bfgs_labels = []
    for l_l2sq in l_l2sq_list:
        hawkes_lbgfsb = HawkesSumExpKern(decays, gofit='likelihood',
                                         verbose=True,
                                         C=1 / l_l2sq, penalty='l2',
                                         solver='l-bfgs-b', tol=0,
                                         record_every=1, max_iter=max_iter)
        hawkes_lbgfsb._model_obj.n_threads = 4

        hawkes_lbgfsb._solver_obj.history._history_func['estimation err'] = \
            lambda x, **kwargs: estimation_error(x[n_nodes:],
                                                 hawkes_exp_kernels.adjacency)

        hawkes_lbgfsb.fit(timestamps)

        hawkes_bfgs_learners += [hawkes_lbgfsb]
        if len(l_l2sq_list) > 1:
            hawkes_bfgs_labels += ['Hawkes LBFGS {:.2g}'.format(l_l2sq)]
        else:
            hawkes_bfgs_labels += ['L-BFGS-B']

    hawkes_dual_learners = []
    hawkes_dual_labels = []
    for l_l2sq in l_l2sq_list:
        hawkes_dual = HawkesDual(decays, l_l2sq, verbose=True, record_every=1,
                                 n_threads=4, max_iter=max_iter, tol=1e-13,
                                 seed=2309)
        hawkes_dual.history._history_func['estimation err'] = \
            lambda x, **kwargs: estimation_error(x[n_nodes:],
                                                 hawkes_exp_kernels.adjacency)

        hawkes_dual.fit(timestamps)
        hawkes_dual._learner

        hawkes_dual_learners += [hawkes_dual]
    if len(l_l2sq_list) > 1:
        hawkes_dual_labels += ['Hawkes Dual {:.2g}'.format(l_l2sq)]
    else:
        hawkes_dual_labels += ['Shifted SDCA']

    # print(hawkes_lbgfsb._solver_obj.objective(hawkes_dual.solution))
    # print(hawkes_dual.objective(hawkes_dual.solution))

    learners = hawkes_bfgs_learners + hawkes_dual_learners
    labels = hawkes_bfgs_labels + hawkes_dual_labels

    ax_start = n_plot
    plot_hawkes_kernel_norms(hawkes_exp_kernels, node_names=[],
                             ax=ax_list[0][ax_start])

    plot_history(learners, labels=labels, ax=ax_list[1][ax_start],
                 y='estimation err')

    if positive:
        subtitle = 'with no inhibition'
    else:
        subtitle = 'with inhibition'

    ax_list[0][ax_start].set_title('Original kernel norms\n' + subtitle, y=1.05)
    ax_list[1][ax_start].set_title('Estimation error', y=1.02)
    ax_list[1][ax_start].set_xlabel('Passes over data', fontsize=10)
    ax_list[1][ax_start].set_ylabel('')
    ax_list[1][ax_start].set_xlim([0, max_iter - 5])
    ax_list[1][ax_start].set_yscale('symlog', linthreshy=1.)
    ax_list[1][ax_start].set_ylim([0, None])

fig.tight_layout()

plt.show()
