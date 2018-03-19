import os
from collections import OrderedDict
from itertools import product

from joblib import Parallel, delayed

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero
from tick.optim.solver import SDCA


ROOT_FOLDER_RESULT = 'dual_viz_results'


def make_file_path(dataset, l_l2sq, filename):
    return os.path.join(ROOT_FOLDER_RESULT, dataset,
                        'l={:.4g}'.format(l_l2sq), '{}.pkl'.format(filename))


def load_experiments(dataset, l_l2sq, filename):
    file_path = make_file_path(dataset, l_l2sq, filename)
    with open(file_path, 'rb') as read_file:
        return pickle.load(read_file)


def save_experiments(experiments, dataset, l_l2sq, filename):
    file_path = make_file_path(dataset, l_l2sq, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as write_file:
        pickle.dump(experiments, write_file)


def run_solver(dataset, features, labels, l_l2sq, max_iter_sdca=1000,
               init_type='psi'):
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    tol = 1e-16
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=max(1, int(max_iter_sdca / 7)), tol=tol,
                batch_size=2, verbose=True)
    sdca.set_model(model).set_prox(ProxZero())
    dual_inits = model.get_dual_init(l_l2sq, init_type)
    sdca._solver.set_starting_iterate(dual_inits)

    sdca.solve()
    save_solver(dataset, sdca)
    return sdca

def save_solver(dataset, sdca):
    history = sdca.history
    l_l2sq = sdca.l_l2sq
    save_experiments(history, dataset, l_l2sq, 'history')

    duality_gap = history.last_values['duality_gap']
    if duality_gap < 1e-5:
        dual_at_optimum = history.last_values['dual_vector']
        save_experiments(dual_at_optimum, dataset, l_l2sq, 'duals')


def compute_slope_init_dual(dataset, features, labels, l_l2sq):
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    duals = load_experiments(dataset, l_l2sq, 'duals')
    dual_inits = model.get_dual_init(l_l2sq)

    non_zero_features = features[labels != 0]
    norms = np.linalg.norm(non_zero_features, axis=1)

    normalized_inits = dual_inits / norms
    normalized_duals = duals / norms

    slope, intercept, r_value, p_value, std_err = \
        linregress(normalized_inits, normalized_duals)

    return slope


def plot_experiment(dataset, l_l2sq, axes=None):
    features, labels = load_dataset(dataset)
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    duals = load_experiments(dataset, l_l2sq, 'duals')

    dual_inits = model.get_dual_init(l_l2sq)

    if axes is None:
        fig, axes = plt.subplots(2, 1)

    non_zero_features = features[labels != 0]
    norms = np.linalg.norm(non_zero_features, axis=1)

    print('duals', duals.shape, norms.shape, sum(labels != 0))
    normalized_inits = dual_inits / norms
    normalized_duals = duals / norms

    slope, intercept, r_value, p_value, std_err = \
        linregress(normalized_inits, normalized_duals)

    if isinstance(axes, np.ndarray):
        axes[0].scatter(normalized_inits, normalized_duals)
        axes[0].plot([0, max(normalized_inits)],
                     [0, slope * max(normalized_inits)],
                     color='r', label='{:.4g} x'.format(slope))
        axes[0].set_ylabel('dual sol / norm')
        axes[0].set_xlabel('dual init / norm')
        axes[0].legend()

        # axes[1].scatter(dual_inits, duals)
        # axes[1].set_ylabel('dual sol')
        # axes[1].set_xlabel('dual init')
        history = load_experiments(dataset, l_l2sq, 'history')
        n_iter = history.values['n_iter']
        dual_history = history.values['dual_vector']
        dual_means = np.array([np.mean(dual_vector)
                               for dual_vector in dual_history])
        dual_norms_diffs = np.array([
            np.linalg.norm(dual_vector - dual_history[-1], 1) / len(dual_history[-1])
            for dual_vector in dual_history])
        dual_means_diff = np.abs(dual_means - dual_means[-1])
        axes[1].plot(n_iter, dual_means_diff, label='dual mean diff')
        axes[1].plot(n_iter, dual_norms_diffs, label='dual norm diff')
        axes[1].set_yscale('log')
        axes[1].legend()

    else:
        axes.scatter(normalized_inits, normalized_duals)
        axes.plot([0, max(normalized_inits)],
                     [0, slope * max(normalized_inits)],
                     color='r', label='{:.4g}'.format(slope))
        axes.set_ylabel('dual sol / norm')
        axes.set_xlabel('dual init / norm')
        axes.legend()
        l_l2sq_coeff = l_l2sq * np.sqrt(len(labels))
        axes.set_title(dataset.replace('simulated_', '') + ' {:.3g}'
                       .format(l_l2sq_coeff))


def load_dataset(dataset_name):
    splitted_name = dataset_name.split('_')
    if len(splitted_name) > 1:
        max_n_samples = int(splitted_name[1])
    else:
        max_n_samples = 10000
    if len(splitted_name) > 2:
        max_n_features = int(splitted_name[2])
    else:
        max_n_features = None

    features, labels = fetch_poisson_dataset(splitted_name[0],
                                             n_samples=max_n_samples,
                                             n_features=max_n_features)
    return features, labels


def all_l_l2sq(n_samples, n_coeffs=4):

    lowest_coef = np.log(np.power(n_samples, -0.3))
    return np.logspace(lowest_coef, 1, n_coeffs) / np.sqrt(n_samples)


def run_all_experiments(datasets):
    for dataset in datasets:
        features, labels = load_dataset(dataset)
        l_l2sq_list = all_l_l2sq(len(labels))

        Parallel(n_jobs=1)(delayed(run_solver)(dataset, features, labels, l_l2sq)
                           for l_l2sq in l_l2sq_list)


def extract_l_l2sq_from_dirname(dir_name):
    return float(dir_name.split("=")[1])


def list_all_experiments(dataset):
    result_folder = os.path.join(ROOT_FOLDER_RESULT, dataset)

    l_l2sqs = []
    all_files = [dir_name for dir_name in os.listdir(result_folder)
                 if dir_name.startswith('l=')]

    for result_dir in all_files:
        l_l2sqs += [extract_l_l2sq_from_dirname(result_dir)]

    sorted_files = [x for _, x in sorted(zip(l_l2sqs, all_files),
                                         key=lambda pair: pair[0])]
    return sorted_files


def plot_slopes(dataset, ax=None):
    features, labels = load_dataset(dataset)

    l_l2sq_list = [extract_l_l2sq_from_dirname(experiment_dir)
                   for experiment_dir in list_all_experiments(dataset)]

    slopes = []
    for l_l2sq in l_l2sq_list:
        slopes += [compute_slope_init_dual(dataset, features, labels, l_l2sq)]

    l_l2sq_list = np.array(l_l2sq_list)
    slopes = np.array(slopes)

    slope, intercept, r_value, p_value, std_err = \
        linregress(np.log(l_l2sq_list), np.log(slopes))

    slope_interpolation = np.exp(intercept) * np.power(l_l2sq_list, slope)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(l_l2sq_list, slopes)
    ax.plot(l_l2sq_list, slope_interpolation,
            label='${:.2g} \\lambda^{{{:.2g}}}$'
            .format(np.exp(intercept), slope))
    ax.set_ylabel('dual / init')
    ax.set_xlabel('l_l2sq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(dataset)
    ax.legend()


if __name__ == '__main__':
    n_features_list = [100, 300]
    n_samples_list = [1000, 10000]

    all_datasets = ['simulated_{}_{}'.format(n_samples, n_features)
                    for n_samples, n_features
                    in product(n_samples_list, n_features_list)]

    run_all_experiments(all_datasets)

    # dataset = all_datasets[-1]
    #
    #

    n_rows = len(n_features_list) * 3
    n_cols = int(np.ceil(len(all_datasets) / n_rows)) * 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    count = 0
    for i, dataset in enumerate(all_datasets):
        # plot_slopes(dataset, ax=ax)
        for experiment in list_all_experiments(dataset):
            l_l2sq = extract_l_l2sq_from_dirname(experiment)

            ax = axes.ravel()[count]
            plot_experiment(dataset, l_l2sq, axes=ax)

            count += 1

    plt.gcf().tight_layout()
    plt.show()

    #
    # for i, dataset in enumerate(all_datasets):
    #     if '_' in dataset:
    #         max_n_samples = int(dataset.split('_')[1])
    #
    #     features, labels = fetch_poisson_dataset(dataset.split('_')[0], n_samples=max_n_samples)
    #
    #     result_folder = os.path.join('dual_viz_results', dataset)
    #
    #     l_l2sqs = []
    #     slopes = []
    #     all_files = [file_name for file_name in os.listdir(result_folder)
    #                  if file_name.startswith('l=')]
    #     all_files.sort()
    #     for result_file in all_files:
    #         l_l2sq = float('.'.join(result_file.split("=")[1].split('.')[:-1]))
    #         l_l2sqs += [l_l2sq]
    #         slopes += [init_slope(features, labels, lambda n, l_l2sq=l_l2sq: l_l2sq)]
    #     # plot_experiment(dataset, l_l2sq_func, axes=axes[:, i])
    #     # axes[0, i].set_title('${}$'.format(label))
    #
    #     slopes = [x for _, x in sorted(zip(l_l2sqs, slopes), key=lambda pair: pair[0])]
    #     l_l2sqs.sort()
    #
    #     print('len(labels)', len(labels))
    #
    #
    #     # min_l2sq = 0.3 / np.sqrt(len(labels))
    #     # print(min_l2sq, l_l2sqs)
    #     l_l2sqs = np.array(l_l2sqs)
    #     slopes = np.array(slopes)
    #     # mask = l_l2sqs < min_l2sq
    #     # slopes = slopes[mask]
    #     # l_l2sqs = l_l2sqs[mask]
    #
    #     print(0.7 + np.log10(len(labels)))
    #     print(linregress(np.log(l_l2sqs), np.log(slopes)))
    #
    #     ax = axes.ravel()[i]
    #     ax.plot(l_l2sqs, slopes)
    #     ax.set_ylabel('dual / init')
    #     ax.set_xlabel('l_l2sq')
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.set_title(dataset)
    #
    # plt.show()
    # # plot_all_last_experiment(all_datasets, l_l2sq_coef=l_l2sq_coef,
    #                          fit_intercept=fit_intercept)
