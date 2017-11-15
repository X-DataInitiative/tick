import tick

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems


def run_solvers(model, l_l2sq, skip_newton=False):
    solvers = []
    coeff0 = np.ones(model.n_coeffs)

    # model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=False)
    # model_dual.fit(features, labels)
    # max_iter_dual_bfgs = 1000
    # lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
    #                      print_every=int(max_iter_dual_bfgs / 7))
    # lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
    # lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
    # print(lbfgsb_dual.solution.mean())
    # print(model_dual.get_primal(lbfgsb_dual.solution))
    # for i, x in enumerate(lbfgsb_dual.history.values['x']):
    #      primal = lbfgsb._proj.call(model_dual.get_primal(x))
    #      lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)
    #

    max_iter_sdca = 1000
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=int(max_iter_sdca / 7), tol=1e-10, batch_size=1)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.solve()
    sdca.history.name = 'SDCA 1'
    solvers += [sdca]

    print(sdca.solution)

    # sdca_2 = SDCA(l_l2sq, max_iter=max_iter_sdca,
    #               print_every=int(max_iter_sdca / 7), tol=1e-10, batch_size=2)
    # sdca_2.set_model(model).set_prox(ProxZero())
    # sdca_2.solve()
    # sdca_2.history.name = 'SDCA 2'
    # solvers += [sdca_2]
    #
    # lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=1e-10)
    # lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    # lbfgsb.solve(coeff0)
    # solvers += [lbfgsb]
    #
    # svrg = SVRG(max_iter=100, print_every=10, tol=1e-10, step=1e-1)
    # svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    # svrg.solve(coeff0)
    # solvers += [svrg]

    sol = np.array([2.29, 1.16, 1.53, 0.78, 0.84, 0.89, 2.11, 1.05, 2.85, 2.00, 3.05])
    sol -= 0.5

    scpg = SCPG(max_iter=15, print_every=1, tol=0, step=1e6, modified=True)
    scpg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    scpg.solve(sol)
    solvers += [scpg]

    # if not skip_newton:
    #     newton = Newton(max_iter=100, print_every=10)
    #     newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    #     newton.solve(coeff0)
    #     solvers += [newton]

    return [solver.history for solver in solvers]


def load_experiments(file_name='poisson_real_data.pkl'):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as read_file:
            experiments = pickle.load(read_file)
    else:
        experiments = {}
    return experiments


def save_experiments(experiments, file_name='poisson_real_data.pkl'):
    with open(file_name, 'wb') as write_file:
        pickle.dump(experiments, write_file)


def run_experiment(dataset='news', show=True):
    features, labels = fetch_poisson_dataset(dataset,
                                             n_samples=max_n_samples)
    labels[:] = 1

    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    l_2sq_list = [1. / np.sqrt(len(labels))]
    fig, ax_list_list = plt.subplots(2, len(l_2sq_list))
    if len(ax_list_list.shape) == 1:
        ax_list_list = np.array([ax_list_list])

    for i, l_2sq in enumerate(l_2sq_list):
        ax_list = ax_list_list[i]

        skip_newton = dataset in ['blog']
        histories = run_solvers(model, l_2sq, skip_newton=skip_newton)
        #
        # now_formatted = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        #
        # experiments = load_experiments()
        # experiments.setdefault(dataset, {})
        # experiments[dataset].setdefault(l_2sq, {})
        # experiments[dataset][l_2sq][now_formatted] = histories
        # save_experiments(experiments)

        plot_history(histories, dist_min=True, log_scale=True,
                     x='time', ax=ax_list[0])
        ax_list_list[0, i].set_title('$\\lambda = {:.3g}$'.format(l_2sq))

    if show:
        plt.show()


def plot_all_last_experiment(datasets=None):
    experiments = load_experiments()
    if datasets is None:
        datasets = experiments.keys()

    if len(datasets) > 3:
        n_rows = 2
        n_cols = int(np.ceil(len(datasets) / n_rows))
    else:
        n_rows = 1
        n_cols = len(datasets)

    fig, ax_list = plt.subplots(n_rows, n_cols, sharey=True,
                                figsize=(3 * n_cols, 3 * n_rows))

    for i, dataset in enumerate(datasets):
        features, labels = fetch_poisson_dataset(dataset,
                                                 n_samples=max_n_samples)
        n = len(labels)

        print(dataset)

        l_l2sq = 1. / np.sqrt(n)
        all_runs = experiments[dataset][l_l2sq]
        run_times = list(all_runs.keys())
        run_times.sort()
        last_run = run_times[-1]

        ax = ax_list.ravel()[i]
        histories = experiments[dataset][l_l2sq][last_run]
        plot_history(histories, dist_min=True, log_scale=True,
                     x='time', ax=ax)

        sdca_index = [history.name for history in histories].index('SDCA 1')
        sdca_time = histories[sdca_index].last_values['time']

        current_lim = ax.get_xlim()
        if sdca_time * 4 < current_lim[1]:
            ax.set_xlim(0, sdca_time * 4)

        ax.set_ylim(1e-10, 1e6)

        ax.set_title('{} $n={}$ $d={}$'.format(
            dataset, features.shape[0], features.shape[1]))

        ax.set_ylabel('')
        if i != len(datasets) - 1:
            ax.legend_.remove()

        position = np.argwhere(ax_list == ax)[0]
        if len(position) > 1:
            row = position[0]
            if row == 0:
                ax.set_xlabel('')

    fig.tight_layout()
    plt.show()


max_n_samples = 10000000
all_datasets = ['wine', 'facebook', 'news', 'crime', 'vegas', 'blog']
# for dataset in all_datasets:
#     run_experiment(dataset, show=False)
run_experiment('wine', show=True)

# plot_all_last_experiment(all_datasets)
