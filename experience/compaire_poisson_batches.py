# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

# from experience.poisreg_sdca import ModelPoisRegSDCA
from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems


def run_solvers(model, l_l2sq, ax_list):
    solvers = []
    # coeff0 = np.ones(model.n_coeffs)

    # lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=1e-10)
    # lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    #
    # model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
    # model_dual.fit(features, labels)
    # max_iter_dual_bfgs = 100
    # lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
    #                      print_every=int(max_iter_dual_bfgs / 7))
    # lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
    # lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
    #
    # lbfgsb_dual.history.values['dual_objective'] = \
    #     [-x for x in lbfgsb_dual.history.values['obj']]
    # for i, x in enumerate(lbfgsb_dual.history.values['x']):
    #      primal = lbfgsb._proj.call(model_dual.get_primal(x))
    #      lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)
    # solvers += [lbfgsb_dual]

    max_iter_sdca = 1000
    sto_seed = 23983

    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-10, seed=sto_seed)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()
    sdca.history.name = 'SDCA #1'
    solvers += [sdca]
    #
    sdca_2_two = SDCA(l_l2sq, max_iter=max_iter_sdca,
                      print_every=int(max_iter_sdca / 7), tol=1e-10,
                      batch_size=2,
                      seed=sto_seed)
    sdca_2_two.set_model(model).set_prox(ProxZero())
    sdca_2_two.solve()
    sdca_2_two.history.name = 'SDCA #2 Ex'
    solvers += [sdca_2_two]

    batch_sizes = [2, 3, 7, 15, 20, 30, 35, 50]
    for batch_size in batch_sizes:
        sdca_batch = SDCA(l_l2sq, max_iter=max_iter_sdca,
                          print_every=int(max_iter_sdca / 7), tol=1e-10,
                          batch_size=batch_size + 1, seed=sto_seed)
        sdca_batch.set_model(model).set_prox(ProxZero())
        sdca_batch.solve()
        sdca_batch.history.name = 'SDCA #{}'.format(sdca_batch.batch_size - 1)
        solvers += [sdca_batch]

    labels = []
    for solver in solvers:
        solver.history.values['dual_objective'] = [
            -x if x != float('-inf') else np.nan
            for x in solver.history.values['dual_objective']
        ]

        if isinstance(solver, SDCA):
            time_per_ite = solver.history.last_values['time'] / \
                           solver.history.last_values['n_iter']
            time_per_ite *= 1000
            labels += [
                '{} {:.2g} ms/i'.format(solver.history.name,
                                        time_per_ite)]
        else:
            labels += [solver.__class__.__name__]

    plot_history(solvers, dist_min=True, log_scale=True,
                 x='time', ax=ax_list[0], labels=labels)

    plot_history(solvers, dist_min=True, log_scale=True,
                 x='time', y='dual_objective', ax=ax_list[1], labels=labels)


dataset = 'crime'
features, labels = fetch_poisson_dataset(dataset, n_samples=10000)

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_2sq_list = [1e-2, 1e-3, 1e-4, 1. / np.sqrt(len(labels))]
l_2sq_list = [1e-4,]
fig, ax_list = plt.subplots(2, len(l_2sq_list), figsize=(12, 8))
if len(l_2sq_list) == 1:
    ax_list = np.array([ax_list]).T

for i, l_2sq in enumerate(l_2sq_list):
    run_solvers(model, l_2sq, ax_list[:, i])
    ax_list[0, i].set_title('$n={}$ $d={}$ $\\lambda = {:.3g}$'
                            .format(features.shape[0], features.shape[1],
                                    l_2sq))
    if i != len(l_2sq_list) - 1:
        ax_list[0, i].legend_.remove()
        ax_list[1, i].legend_.remove()

fig.tight_layout()
plt.show()
