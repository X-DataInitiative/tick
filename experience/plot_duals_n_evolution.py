import os
import warnings
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


ROOT_FOLDER_RESULT = 'dual_n_evolution_results'


def make_file_path(dataset, n_samples, filename, sample_seed):
    return os.path.join(ROOT_FOLDER_RESULT, dataset,
                        'n={}'.format(n_samples),
                        'seed={}'.format(sample_seed),
                        '{}.pkl'.format(filename))


def load_experiments(dataset, n_samples, filename):
    seed_list = list_all_seed_experiments(dataset, n_samples)

    for seed in seed_list:
        file_path = make_file_path(dataset, n_samples, filename, seed)
        with open(file_path, 'rb') as read_file:
            yield seed, pickle.load(read_file)


def save_experiments(experiments, dataset, n_samples, filename, sample_seed):
    file_path = make_file_path(dataset, n_samples, filename, sample_seed)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as write_file:
        pickle.dump(experiments, write_file)


def run_solver(dataset, features, labels, l_l2sq, sample_seed, max_iter_sdca=1000):
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    tol = 1e-10
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=max(1, int(max_iter_sdca / 7)), tol=tol,
                batch_size=2, verbose=True)
    sdca.set_model(model).set_prox(ProxZero())
    dual_inits = model.get_dual_init(l_l2sq)
    sdca._solver.set_starting_iterate(dual_inits)

    sdca.solve()
    save_solver(dataset, sdca, sample_seed)
    return sdca


def save_solver(dataset, sdca, sample_seed):
    history = sdca.history
    n_samples = sdca.model.n_samples
    # save_experiments(history, dataset, n_samples, 'history', sample_seed)

    duality_gap = history.last_values['duality_gap']
    if duality_gap < 1e-3:
        dual_at_optimum = history.last_values['dual_vector']
        save_experiments(dual_at_optimum, dataset, n_samples, 'duals',
                         sample_seed)
    else:
        warnings.warn('Bad duality gap for {} ({})'
                      .format(dataset, duality_gap))


def plot_experiment_n_samples(dataset, n_samples_list, ax, position):
    features, labels = load_dataset(dataset)
    # model = ModelPoisReg(fit_intercept=False, link='identity')
    # model.fit(features, labels)

    mean_duals = []
    max_duals = []
    std_max_duals = []
    n_seeds_used = []
    for n_samples in n_samples_list:
        mean_dual_n_sample = []
        max_dual_n_sample = []
        for seed, duals in load_experiments(dataset, n_samples, 'duals'):
            mean_dual_n_sample += [duals.mean()]
            max_dual_n_sample += [duals.max()]
            # print(len(duals), n_samples, 'seed', seed, 'max dual', duals.max())

        mean_duals += [np.mean(mean_dual_n_sample)]
        max_duals += [np.mean(max_dual_n_sample)]
        std_max_duals += [np.std(max_dual_n_sample)]
        n_seeds_used += [len(max_dual_n_sample)]

    max_duals = np.array(max_duals)
    std_max_duals = np.array(std_max_duals)
    n_seeds_used = np.array(n_seeds_used)

    half_width_confidence = 1.96 * std_max_duals / np.sqrt(n_seeds_used)
    conf_int_upper_bound = max_duals + half_width_confidence
    conf_int_lower_bound = max_duals - half_width_confidence

    ax.plot(n_samples_list, max_duals)
    ax.plot(n_samples_list, conf_int_upper_bound, c='C1', ls='--')
    ax.plot(n_samples_list, conf_int_lower_bound, c='C1', ls='--')

    if position[0] > 0:
        ax.set_xlabel(r'$n$')

    if position[1] == 0:
        ax.set_ylabel(r'$\max_i \alpha_i^*$')

    # ax.set_ylim([0, None])

    ax.set_title('{} $n={}$ $d={}$'.format(
        dataset, features.shape[0], features.shape[1]))


def load_dataset(dataset_name):
    splitted_name = dataset_name.split('_')
    if len(splitted_name) > 1:
        max_n_samples = int(splitted_name[1])
    else:
        max_n_samples = 100000
    if len(splitted_name) > 2:
        max_n_features = int(splitted_name[2])
    else:
        max_n_features = None

    features, labels = fetch_poisson_dataset(splitted_name[0],
                                             n_samples=max_n_samples,
                                             n_features=max_n_features)
    return features, labels


def compute_l_l2sq(features, labels):
    non_zero_features = features[labels != 0]
    n = len(non_zero_features)
    norms = np.linalg.norm(non_zero_features, axis=1)
    mean_features_norm = np.mean(norms) ** 2
    return mean_features_norm / n


def run_all_experiments(datasets, n_samples_coeff_list, sample_seed):
    for dataset in datasets:
        print(dataset)
        features, labels = load_dataset(dataset)
        l_l2sq = compute_l_l2sq(features, labels)

        for n_samples_coeff in n_samples_coeff_list:
            n_selected_samples = int(n_samples_coeff * len(labels))
            np.random.seed(sample_seed)
            selected_samples = np.random.choice(np.arange(len(labels)),
                                                n_selected_samples,
                                                replace=False)

            selected_features = features[selected_samples, :]
            selected_labels = labels[selected_samples]

            print(n_selected_samples)

            run_solver(dataset, selected_features, selected_labels, l_l2sq,
                       sample_seed)


def extract_n_samples_from_dirname(dir_name):
    return int(dir_name.split("n=")[1])


def extract_seed_from_dirname(dir_name):
    return int(dir_name.split("seed=")[1])


def list_all_n_samples_experiments(dataset):
    result_folder = os.path.join(ROOT_FOLDER_RESULT, dataset)

    n_samples_list = []
    all_files = [dir_name for dir_name in os.listdir(result_folder)
                 if dir_name.startswith('n=')]

    for result_dir in all_files:
        n_samples_list += [extract_n_samples_from_dirname(result_dir)]

    sorted_folders = [x for _, x in sorted(zip(n_samples_list, all_files),
                                         key=lambda pair: pair[0])]
    return sorted_folders

def list_all_seed_experiments(dataset, n_samples):
    result_folder = os.path.join(ROOT_FOLDER_RESULT, dataset,
                                 'n={}'.format(int(n_samples)))

    seed_list = []
    all_files = [dir_name for dir_name in os.listdir(result_folder)
                 if dir_name.startswith('seed=')]

    for result_dir in all_files:
        seed_list += [extract_seed_from_dirname(result_dir)]

    return seed_list


if __name__ == '__main__':

    datasets = ['wine', 'facebook', 'news', 'vegas', 'property', 'simulated']
    datasets = ['simulated']


    n_samples_coeff_list = np.linspace(0.1, 1, 10) #[0.1, 0.3, 0.7, 1]

    for seed in map(int, np.arange(0, 10, 1)):
        print('\nSEED', seed)
        run_all_experiments(datasets, n_samples_coeff_list, seed)

    if len(datasets) > 3:
        n_rows = 2
        n_cols = int(np.ceil(len(datasets) / n_rows))
    else:
        n_rows = 1
        n_cols = len(datasets)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 2.5 * n_rows))
    axes = np.array(axes).reshape((n_rows, n_cols))

    count = 0
    for i, dataset in enumerate(datasets):

        n_samples_list = [
            extract_n_samples_from_dirname(experiment)
            for experiment in list_all_n_samples_experiments(dataset)
        ]

        ax = axes.ravel()[count]
        position = np.argwhere(axes == ax)[0]

        plot_experiment_n_samples(dataset, n_samples_list, ax=ax,
                                  position=position)

        count += 1

    plt.gcf().tight_layout()
    plt.show()
