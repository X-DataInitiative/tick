import tick

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

from experience.poisreg_sdca import ModelPoisRegSDCA
from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero, ProxL2Sq, ProxPositive
from tick.optim.solver import SDCA, Newton, LBFGSB, SVRG, SCPG
from tick.plot import plot_history, stems


def run_solvers(model, l_l2sq, skip_newton=False):
    solvers = []
    coeff0 = np.ones(model.n_coeffs)

    tol = 0
    max_iter_sdca = 100
    sdca = SDCA(l_l2sq, max_iter=100,
                print_every=max(1, int(max_iter_sdca / 7)), tol=tol,
                batch_size=2, verbose=False)
    sdca.set_model(model).set_prox(ProxZero())

    sdca.solve()

    sdca.history.name = 'SDCA 2'
    solvers += [sdca]

    # batch_sizes = [10, 30]
    # for batch_size in batch_sizes:
    #     sdca_batch = SDCA(l_l2sq, max_iter=max_iter_sdca,
    #                       print_every=int(max_iter_sdca / 7), tol=1e-10,
    #                       batch_size=batch_size + 1)
    #     sdca_batch.set_model(model).set_prox(ProxZero())
    #     sdca_batch.solve()
    #     sdca_batch.history.name = 'SDCA #{}'.format(sdca_batch.batch_size - 1)
    #     solvers += [sdca_batch]

    # lbfgsb = LBFGSB(max_iter=100, print_every=10, tol=tol)
    # lbfgsb.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    # lbfgsb.solve(coeff0)
    # solvers += [lbfgsb]

    # model_dual = ModelPoisRegSDCA(l_l2sq, fit_intercept=fit_intercept)
    # model_dual.fit(model.features, model.labels)
    # max_iter_dual_bfgs = 100
    # lbfgsb_dual = LBFGSB(tol=1e-10, max_iter=max_iter_dual_bfgs,
    #                      print_every=int(max_iter_dual_bfgs / 7))
    # lbfgsb_dual.set_model(model_dual).set_prox(ProxPositive())
    # lbfgsb_dual.history.name = 'LBFGS dual'
    # lbfgsb_dual.solve(0.2 * np.ones(model_dual.n_coeffs))
    # for i, x in enumerate(lbfgsb_dual.history.values['x']):
    #     primal = lbfgsb._proj.call(model_dual.get_primal(x))
    #     lbfgsb_dual.history.values['obj'][i] = lbfgsb.objective(primal)
    # solvers += [lbfgsb_dual]

    # svrg = SVRG(max_iter=100, print_every=10, tol=tol, step=1e-3, step_type='bb')
    # svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    # svrg.solve(coeff0)
    # solvers += [svrg]

    # scpg = SCPG(max_iter=15, print_every=1, tol=0, step=1e6, modified=True)
    # scpg.set_model(model).set_prox(ProxL2Sq(l_l2sq, positive=True))
    # scpg.solve(sol)
    # solvers += [scpg]

    # if not skip_newton:
    #     newton = Newton(max_iter=100, print_every=10, tol=tol)
    #     newton.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    #     newton.solve(coeff0)
    #     solvers += [newton]
    #
    # sdca_sol = sdca.solution.copy()
    # arg_sort = sdca_sol.argsort()
    # print(arg_sort[:10])
    # print(sdca.solution[arg_sort[:10]])

    return [solver for solver in solvers]


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


def make_key(l_l2sq, fit_intercept):
    return '{:.4g}'.format(l_l2sq), fit_intercept


def run_experiment(dataset='news', show=True, l_l2sq_coef=1., fit_intercept=False):
    features, labels = fetch_poisson_dataset(dataset,
                                             n_samples=max_n_samples)
    model = ModelPoisReg(fit_intercept=fit_intercept, link='identity')
    model.fit(features, labels)

    l_2sq_list = [l_l2sq_coef / np.sqrt(len(labels))]

    if show:
        fig, ax_list_list = plt.subplots(2, len(l_2sq_list))
        if len(ax_list_list.shape) == 1:
            ax_list_list = np.array([ax_list_list])

    for i, l_l2sq in enumerate(l_2sq_list):

        skip_newton = dataset in ['blog']
        histories = run_solvers(model, l_l2sq, skip_newton=skip_newton)

        now_formatted = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

        # experiments = load_experiments()
        # experiments.setdefault(dataset, {})
        # experiments[dataset].setdefault(make_key(l_l2sq, fit_intercept), {})
        # experiments[dataset][make_key(l_l2sq, fit_intercept)][now_formatted] = histories
        # save_experiments(experiments)
        #
        # print('save', dataset, make_key(l_l2sq, fit_intercept), now_formatted)

        if show:
            ax_list = ax_list_list[i]
            plot_history(histories, dist_min=True, log_scale=True,
                         x='n_iter', ax=ax_list[0])
            ax_list_list[0, i].set_title('{} $\\lambda = {:.3g}$'.format(dataset, l_l2sq))

            corr = model.labels[labels != 0] / \
                   (features[labels != 0] * features[labels != 0]).sum(axis=1)
            print(corr)
            # ax_list[1].hist(histories[0].last_values['dual_vector'], 30)
            duals = histories[0].dual_solution
            # duals /= labels[labels != 0]
            dual_inits = model.get_dual_init(l_l2sq)
            # dual_inits /= labels[labels != 0]

            # mask = duals > 10
            # ax_list[1].hist(duals[mask] - dual_inits[mask], 100)
            # ax_list[1].hist(dual_inits, 100)
            ax_list[1].scatter(dual_inits, duals)
            ax_list[1].set_ylabel('dual sol')
            ax_list[1].set_xlabel('dual init')
            # biggest_indices = duals.argsort()[-20:][::-1]
            # print('max duals i', biggest_indices)
            # print('max duals', duals[biggest_indices])
            solution = histories[0].solution
            # print('dot prods', features[labels != 0][biggest_indices].dot(solution))
            # print()
            # print(features[labels != 0][biggest_indices])
            # print()
            # print(histories[0].last_values['x'])

            # n_non_zeros = sum(labels != 0)
            # non_zero_features = features[labels != 0]
            # non_zero_labels = labels[labels != 0]
            #
            # w_features_sum = np.sum(non_zero_features * (duals - 1)[:, np.newaxis], axis=0)
            # print(features.shape, w_features_sum.shape)
            # print('duals?')
            # print(l_l2sq * n_non_zeros * non_zero_labels / (non_zero_features.dot(w_features_sum)))

            labels = model.labels
            features = model.features
            n_non_zeros = sum(labels != 0)
            non_zero_features = features[labels != 0]

            n_psi = np.sum(features, axis=0)
            n_psi_x = n_psi.dot(non_zero_features.T)

            w_features_sum = np.sum(
                non_zero_features * duals[:, np.newaxis], axis=0)

            sol = 1 / (l_l2sq * n_non_zeros) * (w_features_sum - n_psi)
            print('should be close to 0')
            print(solution[:10])
            print(sol[:10])
            print(model._sdca_primal_dual_relation(l_l2sq, duals)[:10])
            print(duals)

            # ax_list[1].plot([0, 0.28], [0, 50], c='r')

    if show:
        plt.show()
        pass


def plot_all_last_experiment(datasets=None, l_l2sq_coef=1., fit_intercept=False):
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


        l_l2sq = l_l2sq_coef / np.sqrt(n)
        all_runs = experiments[dataset][make_key(l_l2sq, fit_intercept)]
        run_times = list(all_runs.keys())
        run_times.sort()
        last_run = run_times[-1]

        print(dataset, last_run)

        ax = ax_list.ravel()[i]
        histories = experiments[dataset][make_key(l_l2sq, fit_intercept)][last_run]
        plot_history(histories, dist_min=True, log_scale=True,
                     x='time', ax=ax)

        # print([history.name for history in histories])
        sdca_index = [history.name for history in histories].index('SDCA 2')
        sdca_time = histories[sdca_index].last_values['time']

        current_lim = ax.get_xlim()
        if sdca_time * 4 < current_lim[1]:
            ax.set_xlim(0, sdca_time * 4)

        ax.set_ylim(1e-13, 1e6)

        ax.set_title('{} $n={}$ $d={}$'.format(
            dataset, features.shape[0], features.shape[1]))

        ax.set_ylabel('')
        ax.legend_.remove()

        position = np.argwhere(ax_list == ax)[0]
        if len(position) > 1:
            row = position[0]
            if row == 0:
                ax.set_xlabel('')

    fig.suptitle('$\\lambda = \\frac{{{}}} {{ \\sqrt{{n}}}}$ {} intercept'
                 .format(l_l2sq_coef, 'with' if fit_intercept else 'without'),
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    handles, labels = ax_list.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    plt.show()


fit_intercept = False
l_l2sq_coef = 1
max_n_samples = 100000
# all_datasets = ['wine', 'facebook', 'crime', 'vegas', 'news', 'blog']
all_datasets = ['crime', 'facebook', 'crime', 'vegas']
# all_datasets = ['facebook', 'blog']
# all_datasets = ['wine', 'blog']


# for dataset in all_datasets:
#     run_experiment(dataset, show=False, l_l2sq_coef=l_l2sq_coef,
#                    fit_intercept=fit_intercept)
run_experiment('news', show=True, fit_intercept=fit_intercept,
               l_l2sq_coef=l_l2sq_coef)


# plot_all_last_experiment(all_datasets, l_l2sq_coef=l_l2sq_coef,
#                          fit_intercept=fit_intercept)
