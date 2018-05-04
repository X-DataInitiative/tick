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


def run_solvers(model, l_l2sq, max_iter_sdca):
    solvers = []

    sto_seed = 23983

    dual_init = model.get_dual_init(l_l2sq)

    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-8, seed=sto_seed,
                store_only_x=True)
    sdca.set_model(model).set_prox(ProxZero())
    sdca._solver.set_starting_iterate(dual_init)
    sdca.history.name = 'SDCA #1'
    sdca.solve()
    solvers += [sdca]
    #
    sdca_2_two = SDCA(l_l2sq, max_iter=max_iter_sdca,
                      print_every=int(max_iter_sdca / 7), tol=1e-13,
                      batch_size=2,
                      seed=sto_seed, store_only_x=True)
    sdca_2_two.set_model(model).set_prox(ProxZero())
    sdca_2_two._solver.set_starting_iterate(dual_init)
    sdca_2_two.history.name = 'SDCA #2'
    sdca_2_two.solve()
    solvers += [sdca_2_two]

    batch_sizes = [2, 10]
    for batch_size in batch_sizes:
        if batch_size == 2:
            iter_sdca = max_iter_sdca
        else:
            iter_sdca = int(max_iter_sdca * 8 / batch_size)

        sdca_batch = SDCA(l_l2sq, max_iter=iter_sdca,
                          print_every=max(int(iter_sdca / 7), 1), tol=1e-10,
                          batch_size=batch_size + 1, seed=sto_seed,
                          store_only_x=True)
        sdca_batch.set_model(model).set_prox(ProxZero())
        sdca_batch._solver.set_starting_iterate(dual_init)
        sdca_batch.history.name = 'SDCA #{} blas'.format(sdca_batch.batch_size - 1)
        sdca_batch.solve()
        solvers += [sdca_batch]

    save_folder = os.path.join(result_folder_path, dataset)
    os.makedirs(save_folder, exist_ok=True)
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
            # label = \
            #     '{} {:.2g} ms/i'.format(solver.history.name,
            #                             time_per_ite)
            label = solver.history.name
        else:
            label = solver.__class__.__name__

        solver_dict = {
            'info': {
                'l_l2_sq': l_l2sq, 'batch_size': solver.batch_size,
                'label': label},
            'history': solver.history
        }

        with open(os.path.join(save_folder,
                               '{}.pkl'.format(solver.history.name)), 'wb') \
                as output_file:
            pickle.dump(solver_dict, output_file)


def load_results(dataset):
    data_result_folder_path = os.path.join(result_folder_path, dataset)
    labels = []
    histories = []
    dual_objectives = []
    batch_sizes = []

    for file_name in os.listdir(data_result_folder_path):
        if file_name.endswith('pkl'):
            res_file = os.path.join(data_result_folder_path, file_name)
            with open(res_file, 'rb') as input_file:
                solver_dict = pickle.load(input_file)

            histories += [solver_dict['history']]
            dual_objectives += [histories[-1].values['dual_objective']]
            batch_sizes += [solver_dict['info']['batch_size']]
            labels += [solver_dict['info']['label']]

    batch_sizes = np.array(batch_sizes)
    batch_sizes_order = np.argsort(batch_sizes)

    batch_sizes = batch_sizes[batch_sizes_order]
    labels = [labels[i] for i in batch_sizes_order]
    dual_objectives = [dual_objectives[i] for i in batch_sizes_order]
    histories = [histories[i] for i in batch_sizes_order]

    return batch_sizes, labels, dual_objectives, histories


def compute_l_l2sq(features, labels):
    non_zero_features = features[labels != 0]
    n = len(non_zero_features)
    norms = np.linalg.norm(non_zero_features, axis=1)
    mean_features_norm = np.mean(norms) ** 2

    return mean_features_norm / n


def plot_time_per_iteration(dataset, ax):
    batch_sizes, labels, _, histories = load_results(dataset)

    time_per_ites = []
    for history in histories:
        time_per_ite = history.last_values['time'] / \
                       history.last_values['n_iter']
        time_per_ite *= 1000
        time_per_ites += [time_per_ite]

    ax.plot(batch_sizes, time_per_ites)


def plot_time_to_reach_precision(dataset, ax):
    batch_sizes, labels, dual_objectives, histories = load_results(dataset)

    min_dual_objective = np.nanmin(np.hstack(dual_objectives))

    shifted_dual_objectives = [
        np.array(dual_objective) - min_dual_objective
        for dual_objective in dual_objectives]

    precisions = np.logspace(1, -5, 5)
    for precision in precisions:
        precison_iterations = [np.argmax(dual_objective < precision)
                               if np.any(dual_objective < precision)
                               else np.nan
                               for dual_objective in
                               shifted_dual_objectives[2:]
                               ]
        precision_time = [history.values['time'][precison_iteration]
                          if not np.isnan(precison_iteration)
                          else np.nan
                          for precison_iteration, history in
                          zip(precison_iterations, histories[2:])]
        ax.plot(batch_sizes[2:], precision_time, label=precision)

    # ax_list[1].set_yscale("log")
    ax_list[1].legend()


def plot_results(dataset, ax_list):
    batch_sizes, labels, dual_objectives, histories = load_results(dataset)

    plot_time_per_iteration(dataset, ax_list[0])
    plot_time_to_reach_precision(dataset, ax_list[1])

    plot_history(histories, dist_min=True, log_scale=True,
                 x='time', y='dual_objective', ax=ax_list[2], labels=labels)


fit_intercept = False
max_n_samples = 100000

plot_per_dataset = False

datasets = ['wine', 'facebook', 'news', 'vegas', 'property', 'simulated']
max_iter_dict = {
    # 'crime': 1000,
    # 'news': 30,
    'property': 150,
    'simulated': 30,
}

if not plot_per_dataset:
    if len(datasets) > 3:
        n_rows = 2
        n_cols = int(np.ceil(len(datasets) / n_rows))
    else:
        n_rows = 1
        n_cols = len(datasets)

    fig, axes = plt.subplots(n_rows, n_cols, sharey=True,
                                figsize=(3 * n_cols, 2.5 * n_rows))

for i, dataset in enumerate(datasets):

    print()
    print('-' * 50)
    print(' ' * 20, dataset)
    print('-' * 50)

    if dataset == 'simulated':
        features, labels = fetch_poisson_dataset(dataset, n_samples=50000,
                                                 n_features=1000)
    else:
        features, labels = fetch_poisson_dataset(dataset,
                                                 n_samples=max_n_samples)

    model = ModelPoisReg(fit_intercept=fit_intercept, link='identity')
    model.fit(features, labels)

    l_l2sq = compute_l_l2sq(features, labels)

    if dataset in ['simulated']:
        run_solvers(model, l_l2sq, max_iter_dict.get(dataset, 20))

    if plot_per_dataset :
        fig = plt.figure(figsize=(10, 6))
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        ax_list = [ax1, ax2, ax3]
        plot_results(dataset, ax_list)

        fig.tight_layout()
        plt.show()

    else:
        batch_sizes, labels, dual_objectives, histories = load_results(dataset)

        # for label, history in zip(labels, histories):
        #     print(label, history.values['time'][:5])

        ax = axes.ravel()[i]
        plot_history(histories, dist_min=True, log_scale=True,
                     x='time', y='dual_objective', ax=ax,
                     labels=labels)

        ax.set_title('{} $n={}$ $d={}$'.format(
            dataset, features.shape[0], features.shape[1]))

        ax.set_ylabel('')
        ax.legend_.remove()
        ax.set_ylim(1e-12, 1e0)

        position = np.argwhere(axes == ax)[0]
        if len(position) > 1:
            row = position[0]
            if row == 0:
                ax.set_xlabel('')

if not plot_per_dataset:
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

fig.tight_layout()
plt.show()
