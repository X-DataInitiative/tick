
import datetime
import os
from collections import OrderedDict

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np

from experience.poisson_datasets import fetch_poisson_dataset
from tick.optim.model import ModelPoisReg
from tick.optim.prox import ProxZero
from tick.optim.solver import SDCA
from tick.plot import plot_history
from scipy.stats import linregress


def make_file_path(dataset, l_l2sq):
    return os.path.join('dual_viz_results', dataset,
                        'l={:.4g}.pkl'.format(l_l2sq))


def load_experiments(dataset, l_l2sq):
    file_path = make_file_path(dataset, l_l2sq)
    with open(file_path, 'rb') as read_file:
        return pickle.load(read_file)


def save_experiments(experiments, dataset, l_l2sq):
    file_path = make_file_path(dataset, l_l2sq)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as write_file:
        pickle.dump(experiments, write_file)


def run_solver(dataset, features, labels, l_l2sq_func):
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    l_l2sq = l_l2sq_func(len(labels))

    tol = 1e-16
    max_iter_sdca = 1000
    sdca = SDCA(l_l2sq, max_iter=max_iter_sdca,
                print_every=max(1, int(max_iter_sdca / 7)), tol=tol,
                batch_size=2, verbose=True)
    sdca.set_model(model).set_prox(ProxZero())
    # dual_init = model.get_dual_init(l_l2sq)
    # sdca._solver.set_starting_iterate(dual_init)
    sdca.solve()

    save_experiments(sdca.history, dataset, l_l2sq)

    # plot_history([sdca], dist_min=True, log_scale=True, x='n_iter')
    return sdca

def init_slope(features, labels, l_l2sq_func):
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    l_l2sq = l_l2sq_func(len(labels))
    history = load_experiments(dataset, l_l2sq)

    duals = history.values['dual_vector'][-1]
    dual_inits = model.get_dual_init(l_l2sq)

    non_zero_features = features[labels != 0]
    norms = np.linalg.norm(non_zero_features, axis=1)

    normalized_inits = dual_inits / norms
    normalized_duals = duals / norms

    slope, intercept, r_value, p_value, std_err = \
        linregress(normalized_inits, normalized_duals)

    return slope


def plot_experiment(dataset, l_l2sq_func, axes=None):
    features, labels = fetch_poisson_dataset(dataset, n_samples=max_n_samples)
    model = ModelPoisReg(fit_intercept=False, link='identity')
    model.fit(features, labels)

    l_l2sq = l_l2sq_func(len(labels))
    history = load_experiments(dataset, l_l2sq)

    duals = history.values['dual_vector'][-1]
    # duals /= labels[labels != 0]
    dual_inits = model.get_dual_init(l_l2sq)
    print('dual_inits', dual_inits.shape)
    # dual_inits /= labels[labels != 0]

    if axes is None:
        fig, axes = plt.subplots(2, 1)

    non_zero_features = features[labels != 0]
    norms = np.linalg.norm(non_zero_features, axis=1)

    normalized_inits = dual_inits / norms
    normalized_duals = duals / norms

    slope, intercept, r_value, p_value, std_err = \
        linregress(normalized_inits, normalized_duals)

    axes[0].scatter(normalized_inits, normalized_duals)
    axes[0].plot([0, max(normalized_inits)],
                 [0, slope * max(normalized_inits)],
                 color='r', label='{:.4g} x'.format(slope))
    axes[0].legend()

    axes[1].scatter(dual_inits, duals)
    axes[1].set_ylabel('dual sol')
    axes[1].set_xlabel('dual init')


max_n_samples = 100000
# all_datasets = ['wine', 'facebook', 'crime', 'vegas', 'news', 'blog']
# all_datasets = ['crime', 'facebook', 'news', 'vegas']
# all_datasets = ['facebook', 'blog']
# all_datasets = ['wine', 'blog']
all_datasets = ['simulated_100', 'simulated_300', 'simulated_1000',
                'simulated_3000', 'simulated_10000', 'simulated_30000']

# for dataset in all_datasets:
#     run_experiment(dataset, show=False, l_l2sq_coef=l_l2sq_coef,
#                    fit_intercept=fit_intercept)


# l_l2sq_funcs['1 / n'] = lambda n: 1 / n
# l_l2sq_funcs['0.01 / \\sqrt{{n}}'] = lambda n: 0.01 / np.sqrt(n)
# l_l2sq_funcs['1 / \\sqrt{{n}}'] = lambda n: 1 / np.sqrt(n)
# l_l2sq_funcs['100 / \\sqrt{{n}}'] = lambda n: 100 / np.sqrt(n)

fig, axes = plt.subplots(2, 3)
for i, dataset in enumerate(all_datasets):
    if '_' in dataset:
        max_n_samples = int(dataset.split('_')[1])

    features, labels = fetch_poisson_dataset(dataset.split('_')[0],
                                             n_samples=max_n_samples)

    l_l2sq_funcs = OrderedDict()
    lowest_coef = np.log(np.power(len(labels), -0.3))
    for l_l2sq_coef in np.logspace(lowest_coef, 0, 15):
        l_l2sq_funcs['{} / \\sqrt{{n}}'.format(l_l2sq_coef)] = \
            lambda n, l_l2sq_coef=l_l2sq_coef: l_l2sq_coef / np.sqrt(n)

    #
    # fig, axes = plt.subplots(2, len(l_l2sq_funcs), figsize=(10, 5))
    # axes = axes.reshape(2, len(l_l2sq_funcs))

    for i, (label, l_l2sq_func) in enumerate(l_l2sq_funcs.items()):
        # run_solver(dataset, features, labels, l_l2sq_func)
        # l_l2sqs += [l_l2sq_func(len(labels))]
        pass

for i, dataset in enumerate(all_datasets):
    if '_' in dataset:
        max_n_samples = int(dataset.split('_')[1])

    features, labels = fetch_poisson_dataset(dataset.split('_')[0], n_samples=max_n_samples)

    result_folder = os.path.join('dual_viz_results', dataset)

    l_l2sqs = []
    slopes = []
    all_files = [file_name for file_name in os.listdir(result_folder)
                 if file_name.startswith('l=')]
    all_files.sort()
    for result_file in all_files:
        l_l2sq = float('.'.join(result_file.split("=")[1].split('.')[:-1]))
        l_l2sqs += [l_l2sq]
        slopes += [init_slope(features, labels, lambda n, l_l2sq=l_l2sq: l_l2sq)]
    # plot_experiment(dataset, l_l2sq_func, axes=axes[:, i])
    # axes[0, i].set_title('${}$'.format(label))

    slopes = [x for _, x in sorted(zip(l_l2sqs, slopes), key=lambda pair: pair[0])]
    l_l2sqs.sort()

    print('len(labels)', len(labels))


    # min_l2sq = 0.3 / np.sqrt(len(labels))
    # print(min_l2sq, l_l2sqs)
    l_l2sqs = np.array(l_l2sqs)
    slopes = np.array(slopes)
    # mask = l_l2sqs < min_l2sq
    # slopes = slopes[mask]
    # l_l2sqs = l_l2sqs[mask]

    print(0.7 + np.log10(len(labels)))
    print(linregress(np.log(l_l2sqs), np.log(slopes)))

    ax = axes.ravel()[i]
    ax.plot(l_l2sqs, slopes)
    ax.set_ylabel('dual / init')
    ax.set_xlabel('l_l2sq')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(dataset)

plt.show()
# plot_all_last_experiment(all_datasets, l_l2sq_coef=l_l2sq_coef,
#                          fit_intercept=fit_intercept)
