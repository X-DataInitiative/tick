"""
==============================
Asynchronous stochastic solver
==============================

This example illustrates the convergence speed of the asynchronous version of
SVRG and SAGA solvers. This solver respectively called KroMagnon and ASAGA
have been introduced in

* Mania, H., Pan, X., Papailiopoulos, D., Recht, B., Ramchandran, K. and Jordan, M.I., 2015.
  Perturbed iterate analysis for asynchronous stochastic optimization.
  `arXiv preprint arXiv:1507.06970.`_.

* R. Leblond, F. Pedregosa, and S. Lacoste-Julien: Asaga: Asynchronous
  Parallel Saga, `(AISTATS) 2017`_.

.. _arXiv preprint arXiv:1507.06970.: https://arxiv.org/abs/1507.06970
.. _(AISTATS) 2017: https://hal.inria.fr/hal-01665255/document

To obtain good speedup in a relative short time example we have designed very
sparse and ill-conditonned problem.
"""
import itertools

import scipy

from scipy import sparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tick.dataset import fetch_tick_dataset
from tick.dataset.url_dataset import fetch_url_dataset
from tick.plot import plot_history
import numpy as np
from tick.linear_model import SimuLogReg, ModelLogReg

from tick.simulation import weights_sparse_gauss
from tick.solver import SVRG, SAGA, SDCA
from tick.prox import ProxElasticNet, ProxL1, ProxL2Sq, ProxZero

from collections import OrderedDict

from tick.linear_model.build.linear_model import ModelLogRegAtomicDouble
from tick.prox.build.prox import ProxL1AtomicDouble
from tick.prox.build.prox import ProxL2SqAtomicDouble

from tick.solver.build.solver import (
    SAGADouble as _SAGADouble,
    AtomicSAGADouble as _ASAGADouble,
    AtomicSAGADoubleAtomicIterate as _ASAGADoubleA,
    SAGADoubleAtomicIterate as _SAGADoubleA,
    AtomicSAGARelax as _ASAGADoubleRelax,
    ExtraAtomicSAGADouble as _ExtraAtomicSAGADouble,
    SVRGDouble as _SVRGDouble,
    SVRGDoubleAtomicIterate as _SVRGDoubleA,
    SDCADouble as _SDCADouble,
    AtomicSDCADouble as _ASDCADouble,
)

import os

from tick.solver.sdca import AtomicSDCA

data_dir = os.path.join("/".join(__file__.split('/')[:-1]),
                        'saved_solvers')


def find_next_folder_name(file_prefix, append=False):
    import os
    os.makedirs(data_dir, exist_ok=True)

    i = 0
    folder_name = os.path.join(data_dir, "{}_{:03}".format(file_prefix, i))
    while os.path.exists(folder_name):
        i += 1
        folder_name = os.path.join(data_dir, "{}_{:03}".format(file_prefix, i))

    if append and i > 0:
        i -= 1
        folder_name = os.path.join(data_dir, "{}_{:03}".format(file_prefix, i))
    return folder_name


def find_last_folder_name(file_prefix):
    import re
    import os

    folder_names = []
    file_regex = "^{}_\d+$".format(file_prefix)
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if re.match(file_regex, folder_name) and os.path.isdir(folder_path):
            folder_names += [folder_path]
    folder_names.sort()
    return folder_names[-1]


def serialize_history(history, solver_class, solver_label, solver_name,
                      n_threads, folder_name, specification):
    import pickle
    # delete space consumming iterates
    if 'x' in history.values.keys():
        del history.values['x']
    if 'dual_vector' in history.values.keys():
        del history.values['dual_vector']
        
    os.makedirs(folder_name, exist_ok=True)
    file_name = os.path.join(folder_name,
                             '{}_{}_threads'.format(solver_class.__name__, n_threads))

    with open(file_name, 'wb') as output_file:
        pickle.dump(dict(history=history, solver_label=solver_label,
                         solver_name=solver_name, specification=specification,
                         n_threads=n_threads),
                    output_file)


def load_histories(file_prefix):
    import pickle
    folder_name = find_last_folder_name(file_prefix)

    specification = None
    histories, solver_labels, solver_names, n_threads_used = [], [], [], []
    for file_name in os.listdir(folder_name):
        with open(os.path.join(folder_name, file_name), 'rb') as input_file:
            infos = pickle.load(input_file)

            histories += [infos['history']]
            solver_labels += [infos['solver_label']]
            solver_names += [infos['solver_name']]
            n_threads_used += [infos['n_threads']]

            # Ensure all histories share the same specifications
            if specification is None:
                specification = infos['specification']
            else:
                assert specification.get('dataset_kwargs', {}) == \
                       infos['specification'].get('dataset_kwargs', {})

    return histories, solver_labels, solver_names, n_threads_used


def compute_l_l2sq(features, labels):
    # return 1. / np.sqrt(len(labels))
    n = len(labels)
    norms = np.power(scipy.sparse.linalg.norm(features, axis=1), 2)
    norms = scipy.sparse.linalg.norm(features, axis=1)
    mean_features_norm = np.mean(norms)

    return mean_features_norm / n


def load_dataset(dataset_name, specification):
    if dataset_name == 'synthetic':
        seed = 1398
        np.random.seed(seed)

        n_samples = specification['n_samples']
        nnz = specification['nnz']
        sparsity = specification['sparsity']
        n_features = int(nnz / sparsity)

        weights = weights_sparse_gauss(n_features, nnz=int(n_features / 3))
        intercept = 0.2
        features = sparse.rand(n_samples, n_features, density=sparsity,
                               format='csr')

        simulator = SimuLogReg(weights, n_samples=n_samples, features=features,
                               verbose=False, intercept=intercept)
        features, labels = simulator.simulate()

    elif dataset_name == 'kdd2010':
        features, labels = fetch_tick_dataset("binary/kdd2010/kdd2010.trn.bz2")

    elif dataset_name == 'reuters':
        features, labels = fetch_tick_dataset("binary/reuters/reuters.trn.bz2")

    elif dataset_name == 'covtype':
        features, labels = fetch_tick_dataset("binary/covtype/covtype.trn.bz2")

    elif dataset_name == 'adult':
        features, labels = fetch_tick_dataset("binary/adult/adult.trn.bz2")

    elif dataset_name == 'rcv1':
        from sklearn.datasets import fetch_rcv1
        rcv1 = fetch_rcv1()
        features = rcv1.data
        # originally rcv1 is multiclass classification but this class is
        # selected in 47% of cases
        labels = (rcv1.target[:, 33] == 1).toarray().reshape(-1).astype(float)
        labels = labels * 2 - 1

    elif dataset_name.startswith('url'):
        n_days = int(dataset_name.split('_')[1])
        features, labels = fetch_url_dataset(n_days=n_days)

    else:
        raise ValueError('Unknown dataset {}'.format(dataset_name))

    if dataset_name != 'synthetic':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        features = scaler.fit_transform(features)

    return features, labels


def create_solver(solver_class, model, penalty_strength, **kwargs):
    ratio = 0.5
    # SDCA "elastic-net" formulation is different from elastic-net
    # implementationl
    l_l2_sdca = ratio * penalty_strength
    l_l1_sdca = (1 - ratio) * penalty_strength
    prox_l1 = ProxL1(l_l1_sdca)
    prox_enet = ProxElasticNet(penalty_strength, ratio)
    
    if solver_class in [_SAGADouble, _ASAGADouble, _ASAGADoubleA,
                        _ASAGADoubleRelax, _SAGADoubleA, _ExtraAtomicSAGADouble,
                        _SVRGDouble, _SVRGDoubleA]:
        step = 1. / model.get_lip_max()

        if solver_class in [_SVRGDouble, _SVRGDoubleA]:
            solver = SVRG(step=step, **kwargs)
        else:
            solver = SAGA(step=step, **kwargs)

        epoch_size = 0
        tol = solver.tol
        _rand_type = solver._rand_type
        step = solver.step
        record_every = solver.record_every
        seed = solver.seed
        n_threads = solver.n_threads

        solver._set('_solver',
                    solver_class(epoch_size, tol, _rand_type, step,
                                 record_every, seed, n_threads))

    elif solver_class == _SDCADouble:
        solver = SDCA(l_l2_sdca, **kwargs)

    elif solver_class == _ASDCADouble:
        solver = AtomicSDCA(l_l2_sdca, **kwargs)
    else:
        raise ValueError('Unknown solver_class {}'.format(solver_class))

    if solver_class in [_SAGADoubleA, _ASAGADoubleA, _SVRGDoubleA, _ExtraAtomicSAGADouble]:
        solver.set_model(model.to_atomic()).set_prox(prox_enet.to_atomic())
    else:
        if solver_class in [_SDCADouble, _ASDCADouble]:
            solver.set_model(model).set_prox(prox_l1)
        else:
            solver.set_model(model).set_prox(prox_enet)

    return solver


def train_model(file_prefix, features, labels, specification, append=False):
    seed = 4320932
    penalty_strength = compute_l_l2sq(features, labels)  # 1e-5
    print('penalty_strength', penalty_strength)

    model = ModelLogReg(fit_intercept=False)
    model.fit(features, labels)
    test_n_threads = specification.get('test_n_threads', [1, 2, 4])

    solver_kwargs = specification['solver_kwargs']
    solver_kwargs.setdefault('max_iter', 10)

    folder_name = find_next_folder_name(file_prefix, append=append)
    for solver_class in classes:
        solver_name = class_names[solver_class]
        print(solver_name)

        for n_threads in test_n_threads:
            print(n_threads)
            solver = create_solver(
                solver_class, model, penalty_strength,
                seed=seed, verbose=False, n_threads=n_threads, tol=0,
                **solver_kwargs)

            solver.solve()

            solver_label = '{} {}'.format(solver_name, n_threads)

            serialize_history(solver.history, solver_class, solver_label,
                              solver_name, n_threads,
                              folder_name, specification)
            
    print('Solution sparsity', np.mean(solver.solution != 0), np.mean(np.abs(solver.solution) < 1e-13))


def plot_histories(file_prefix, ax, specification, keep_threads=None):
    histories, solver_labels, solver_names, n_threads_used = \
        load_histories(file_prefix)

    threads_ls = {1: '-', 2: '--', 4: ':', 8: '-', 16:'--'}

    if keep_threads is not None:
        keep_indices = [i for i in range(len(n_threads_used))
                        if n_threads_used[i] in keep_threads]
        n_threads_used = [n_threads_used[i] for i in keep_indices]
        histories = [histories[i] for i in keep_indices]
        solver_labels = [solver_labels[i] for i in keep_indices]
        solver_names = [solver_names[i] for i in keep_indices]

    plot_history(histories, dist_min=True, log_scale=True,
                 labels=solver_labels, ax=ax, x='time')

    ordered_set_of_solver_names = list(
        OrderedDict.fromkeys(solver_names).keys())
    for line, solver_name, n_threads in zip(ax.lines, solver_names,
                                            n_threads_used):
        color_index = ordered_set_of_solver_names.index(solver_name)
        line.set_color('C{}'.format(color_index))
        line.set_linestyle(threads_ls.get(n_threads, '-'))

    ax.set_ylabel('log distance to optimal objective', fontsize=14)

    fig.tight_layout()
    ax.legend()
    ax.set_ylim([1e-10, None])
    ax.set_title(' '.join(file_prefix.split('_')))


def plot_speedup(file_prefix, ax):
    histories, solver_labels, solver_names, n_threads_used = \
        load_histories(file_prefix)

    solver_names_with_threads = [
        (solver_name, n_threads, index)
        for index, (solver_name, n_threads)
        in enumerate(zip(solver_names, n_threads_used))]
    
    solver_names_with_threads.sort()
    for key, group in itertools.groupby(solver_names_with_threads,
                                        lambda x: x[0]):
        group = list(group)
        group.sort(key=lambda g: g[1])

        group_histories = []
        group_n_threads = []
        for _, _, index in group:
            # print(solver_names[index], n_threads_used[index])
            group_histories += [histories[index]]
            group_n_threads += [n_threads_used[index]]

        solver_objectives = np.array([
            history.values['obj'] for history in group_histories])

        dist_solver_objectives = solver_objectives - solver_objectives.min()

        # ax.plot([test_n_threads[0], test_n_threads[-1]], [1, test_n_threads[-1]],
        #         ls='--', lw=1, c='black')

        for target_precision in [1e-6]:
            target_indexes = [
                np.argwhere(dist_solver_objectives[i] < target_precision)[0][0]
                if dist_solver_objectives[i].min() < target_precision
                else np.nan
                for i in range(len(dist_solver_objectives))
            ]
            print(target_precision, target_indexes)

            target_times = np.array([
                history.values['time'][index]
                if not np.isnan(index)
                else np.nan
                for index, history in zip(target_indexes, group_histories)])

            time_one = target_times[0]
            y = time_one / target_times
            ax.plot(np.array(group_n_threads)[~np.isnan(target_times)],
                    y[~np.isnan(target_times)], marker='x',
                    label=key)
            ax.set_xlabel('number of cores 2')
            ax.set_ylabel('speedup')
            # ax.set_title(solver_name)

        ax.legend()

specifications = {
    'synthetic': {
        'dataset_kwargs': {
            'n_samples': 100000,
            'sparsity': 1e-2,
            'nnz': 50,
        },
        'solver_kwargs': {
            'max_iter': 20,
            'record_every': 2,
        },
        'test_n_threads': [1, 4, 20],
    },
    'rcv1': {
        'solver_kwargs': {
            'max_iter': 60,
            'record_every': 2,
        },
        'test_n_threads': [1, 4, 10] # [25, 30, 35, 40], #[1, 2, 4, 8, 12, 16, 20]#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # [1, 4, 16], #
    },
    'url_20': {
        'solver_kwargs': {
            'max_iter': 60,
            'record_every': 2,
        },
        'test_n_threads': [1, 5, 10, 15, 20], # [1, 4, 16], #
    },
    'url_120': {
        'solver_kwargs': {
            'max_iter': 60,
            'record_every': 2,
        },
        'test_n_threads': [1, 4, 20, 30], # [1, 4, 16], #
    },
    'kdd2010': {
        'solver_kwargs': {
            'max_iter': 60,
            'record_every': 2,
        },
        'test_n_threads': [1, 4, 20, 30], # [1, 4, 16], #
    }
    
}


class_names = {
    _SAGADouble: 'Wild',
    _SAGADoubleA: 'Atomic $w$',
    _ASAGADouble: 'Atomic $\\alpha$',
    _ASAGADoubleA: 'Atomic $w$ and $\\alpha$',
    _ASAGADoubleRelax: 'Atomic $\\alpha$ relax',
    _ExtraAtomicSAGADouble: 'ExAtomic $w$ and $\\alpha$',
    _SVRGDouble: 'Wild',
    _SVRGDoubleA: 'Atomic $w$',
    _SDCADouble: 'Wild',
    _ASDCADouble: 'PassCode',
}

class_series = {
    'SVRG': [_SVRGDouble, _SVRGDoubleA],
    # 'SAGA': [_SAGADouble, _ExtraAtomicSAGADouble, _SAGADoubleA, _ASAGADouble, _ASAGADoubleA],
    'SAGA': [_ExtraAtomicSAGADouble, _ASAGADouble], #[, ],
    #'SAGA': [_ExtraAtomicSAGADouble, _ASAGADouble, _ASAGADoubleA],
    'SDCA': [_ASDCADouble] # , _SDCADouble
}

series = 'SVRG'

if __name__ == '__main__':
    features = None
    for series in ['SVRG']:
        print('\n', series)
        classes = class_series[series]

        dataset_name = 'rcv1'

        specification = specifications[dataset_name]

        file_prefix = '{}_enet_{}'.format(series, dataset_name)

        train = True
        if train:
            
            if features is None:
                features, labels = load_dataset(
                    dataset_name,
                    specification.get('dataset_kwargs', {}))
            # keep = 1000
            # features, labels = features[:keep], labels[:keep]
            print(features.shape, features.nnz / np.prod(features.shape))

            train_model(file_prefix, features, labels, specification,
                        append=False)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        plot_histories(file_prefix, ax, specification, keep_threads=[1, 4, 20])
        # plt.xlim([None, 3])
        # plt.ylim([1e-8, None])
        # plt.show()

        figures_folder = os.path.join("/".join(__file__.split('/')[:-1]), 'figures')
        os.makedirs(figures_folder, exist_ok=True)
        figure_path = os.path.join(figures_folder,
                                   '{}_histories.png'.format(file_prefix))
        print(figure_path)
        plt.savefig(figure_path, dpi=100)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_speedup(file_prefix, ax)
        figure_path = os.path.join(figures_folder,
                                   '{}_speedup.png'.format(file_prefix))
        print(figure_path)
        plt.savefig(figure_path, dpi=100)
