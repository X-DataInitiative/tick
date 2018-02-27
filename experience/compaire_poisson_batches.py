# License: BSD 3 clause

import os
import pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

# from experience.poisreg_sdca import ModelPoisRegSDCA
from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems


result_folder_path = os.path.join(os.path.dirname(__file__), 'batches_results')
os.makedirs(result_folder_path, exist_ok=True)


def run_solvers(model, l_l2sq):
    solvers = []

    max_iter_sdca = 500
    sto_seed = 23983

    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-10, seed=sto_seed,
                store_only_x=True)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()
    sdca.history.name = 'SDCA #1'
    solvers += [sdca]
    #
    sdca_2_two = SDCA(l_l2sq, max_iter=max_iter_sdca,
                      print_every=int(max_iter_sdca / 7), tol=1e-10,
                      batch_size=2,
                      seed=sto_seed, store_only_x=True)
    sdca_2_two.set_model(model).set_prox(ProxZero())
    sdca_2_two.solve()
    sdca_2_two.history.name = 'SDCA #2 Ex'
    solvers += [sdca_2_two]

    batch_sizes = [2, 3, 7, 15, 20, 30, 35, 50, 65, 90, 110]
    for batch_size in batch_sizes:
        sdca_batch = SDCA(l_l2sq, max_iter=max_iter_sdca,
                          print_every=int(max_iter_sdca / 7), tol=1e-10,
                          batch_size=batch_size + 1, seed=sto_seed,
                          store_only_x=True)
        sdca_batch.set_model(model).set_prox(ProxZero())
        sdca_batch.solve()
        sdca_batch.history.name = 'SDCA #{}'.format(sdca_batch.batch_size - 1)
        solvers += [sdca_batch]

    labels = []
    for solver in solvers:
        dual_objectives = [
            solver.dual_objective(dual_vector)
            for dual_vector in solver.history.values['dual_vector']
        ]
        solver.history.values['dual_objective'] = [
            -x if x != float('-inf') else np.nan
            for x in dual_objectives
        ]

        primal_objectives = [
            solver.objective(x) for x in solver.history.values['x']
        ]
        solver.history.values['obj'] = [
            x if x != float('inf') else np.nan
            for x in primal_objectives
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

    save_dict = OrderedDict()
    for label, solver in zip(labels, solvers):
        save_dict[label] = {
            'info': {'l2_sq': l_2sq, 'batch_size': solver.batch_size},
            'history': solver.history
        }

    with open(os.path.join(result_folder_path, dataset), 'wb') as output_file:
        pickle.dump(save_dict, output_file)


def plot_results(dataset, ax_list):
    result_file_path = os.path.join(result_folder_path, dataset)
    with open(result_file_path, 'rb') as input_file:
        result_dict = pickle.load(input_file)

    labels = []
    histories = []
    dual_objectives = []
    batch_sizes = []
    for label in result_dict.keys():
        labels += [label]
        histories += [result_dict[label]['history']]
        dual_objectives += [histories[-1].values['dual_objective']]
        batch_sizes += [result_dict[label]['info']['batch_size']]

    # plot_history(histories, dist_min=True, log_scale=True,
    #              x='time', ax=ax_list[0], labels=labels)

    time_per_ites = []
    for history in histories:
        time_per_ite = history.last_values['time'] / \
                       history.last_values['n_iter']
        time_per_ite *= 1000
        time_per_ites += [time_per_ite]

    ax_list[0].plot(batch_sizes, time_per_ites)

    min_dual_objective = np.nanmin(np.hstack(dual_objectives))

    shifted_dual_objectives = [
        np.array(dual_objective) - min_dual_objective
        for dual_objective in dual_objectives]

    precisions = np.logspace(1, -5, 15)
    for precision in precisions:
        precison_iterations = [np.argmax(dual_objective < precision)
                              if np.any(dual_objective < precision)
                              else np.nan
                              for dual_objective in shifted_dual_objectives[2:]
                              ]
        precision_time = [history.values['time'][precison_iteration]
                          if not np.isnan(precison_iteration)
                          else np.nan
                          for precison_iteration, history in zip(precison_iterations, histories[2:])]
        ax_list[1].plot(batch_sizes[2:], precision_time, label=precision)

    ax_list[1].set_yscale("log")
    # ax_list[1].legend()

    plot_history(histories, dist_min=True, log_scale=True,
                 x='time', y='dual_objective', ax=ax_list[2], labels=labels)


dataset = 'crime'
features, labels = fetch_poisson_dataset(dataset, n_samples=10000)

model = ModelPoisReg(fit_intercept=False, link='identity')
model.fit(features, labels)

l_2sq_list = [1e-2, 1e-3, 1e-4, 1. / np.sqrt(len(labels))]
l_2sq_list = [1e-4,]
# fig, ax_list = plt.subplots(2, len(l_2sq_list), figsize=(12, 8))
# if len(l_2sq_list) == 1:
#     ax_list = np.array([ax_list]).T

for i, l_2sq in enumerate(l_2sq_list):
    # run_solvers(model, l_2sq)

    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax_list = [ax1, ax2, ax3]
    plot_results(dataset, ax_list)

    plt.suptitle('$n={}$ $d={}$ $\\lambda = {:.3g}$'
                            .format(features.shape[0], features.shape[1],
                                    l_2sq))
    # if i != len(l_2sq_list) - 1:
    #     ax_list[0, i].legend_.remove()
    #     ax_list[1, i].legend_.remove()

fig.tight_layout()
plt.show()
