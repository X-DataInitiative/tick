import os

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero
from tick.optim.solver import SDCA
import statsmodels.api as sm

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


def plot_experiment(dataset, l_l2sq, axes=None, scaled=True):
    features, labels = load_dataset(dataset)
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    duals = load_experiments(dataset, l_l2sq, 'duals')

    if scaled:
        dual_inits = model.get_dual_init(l_l2sq, init_type='features')
    else:
        dual_inits = model._get_unscaled_dual_init_features(l_l2sq)

    non_zero_features = features[labels != 0]
    # norms = np.linalg.norm(non_zero_features, axis=1)

    # print('duals', duals.shape, norms.shape, sum(labels != 0))
    # normalized_inits = dual_inits / norms
    # normalized_duals = duals / norms

    # slope, intercept, r_value, p_value, std_err = \
    #     linregress(normalized_inits, normalized_duals)
    linreg_no_interceot = sm.OLS(duals, dual_inits)
    slope_no_intercept = linreg_no_interceot.fit().params[0]

    outlier_mask = (duals / dual_inits > 10) & (duals > 72)
    print('N OUTLIERS', sum(outlier_mask), '/', len(duals))

    print(np.arange(len(duals))[outlier_mask])

    axes.scatter(dual_inits[np.invert(outlier_mask)],
                 duals[np.invert(outlier_mask)],
                 marker='x', alpha=0.5, c='C0')
    axes.scatter(dual_inits[outlier_mask], duals[outlier_mask],
                 marker='x', alpha=0.5, c='C1')

    if scaled:
        smallest_max = min(axes.get_xlim()[1], axes.get_ylim()[1])
        axes.plot([0, smallest_max], [0, smallest_max],
                  color='r', label='$y = x$', ls=':', alpha=0.8)

    else:
        axes.plot([0, max(dual_inits)],
                  [0, slope_no_intercept * max(dual_inits)],
                  color='r', label='$y = {:.2g} x$'.format(slope_no_intercept))

    axes.set_ylabel(r'$\alpha_i^*$')
    if scaled:
        axes.set_xlabel(r'$\alpha_i^{(0)}$')
    else:
        axes.set_xlabel(r'$\kappa_i$')
    axes.legend()
    ax.set_title('{} $n={}$ $d={}$'.format(
        dataset, features.shape[0], features.shape[1]))


def plot_hist_experiment(dataset, l_l2sq, axes=None):
    features, labels = load_dataset(dataset)
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    duals = load_experiments(dataset, l_l2sq, 'duals')

    dual_inits = model.get_dual_init(l_l2sq, )

    if axes is None:
        fig, axes = plt.subplots(2, 1)

    non_zero_features = features[labels != 0]


def load_dataset(dataset_name):
    splitted_name = dataset_name.split('_')
    if len(splitted_name) > 1:
        max_n_samples = int(splitted_name[1])
    else:
        max_n_samples = 10000000
    if len(splitted_name) > 2:
        max_n_features = int(splitted_name[2])
    else:
        max_n_features = None

    features, labels = fetch_poisson_dataset(splitted_name[0],
                                             n_samples=max_n_samples,
                                             n_features=max_n_features)
    return features, labels


def compute_l_l2sq(features, labels):
    # return 1. / np.sqrt(len(labels))
    non_zero_features = features[labels != 0]
    n = len(non_zero_features)
    norms = np.linalg.norm(non_zero_features, axis=1)
    mean_features_norm = np.mean(norms)
    return mean_features_norm / n


def run_all_experiments(datasets):
    for dataset in datasets:
        features, labels = load_dataset(dataset)

        l_l2sq = compute_l_l2sq(features, labels)
        run_solver(dataset, features, labels, l_l2sq)


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


if __name__ == '__main__':

    # datasets = ['wine', 'facebook', 'crime', 'vegas', 'news', 'blog']
    datasets = ['crime']
    scaled = True

    run_all_experiments(datasets)

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
        features, labels = load_dataset(dataset)

        for experiment in list_all_experiments(dataset):
            l_l2sq = extract_l_l2sq_from_dirname(experiment)

            if make_file_path(dataset, l_l2sq, 'x') == \
                    make_file_path(dataset, compute_l_l2sq(features, labels),
                                   'x'):
                ax = axes.ravel()[count]
                plot_experiment(dataset, l_l2sq, axes=ax, scaled=scaled)

                count += 1

    plt.gcf().tight_layout()

    if scaled:
        suffix = 'scaled'
    else:
        suffix = 'unscaled'

    plt.show()
    # plt.savefig(__file__.replace('.py', '_{}.png'.format(suffix)))
