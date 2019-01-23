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
import scipy

from scipy import sparse
import matplotlib.pyplot as plt

from tick.dataset import fetch_tick_dataset
from tick.dataset.url_dataset import fetch_url_dataset
from tick.plot import plot_history
import numpy as np
from tick.linear_model import SimuLogReg, ModelLogReg

from tick.simulation import weights_sparse_gauss
from tick.solver import SVRG, SAGA, SDCA
from tick.prox import ProxElasticNet, ProxL1, ProxZero

from collections import OrderedDict

from tick.linear_model.build.linear_model import ModelLogRegAtomicDouble
from tick.prox.build.prox import ProxL1AtomicDouble

from tick.solver.build.solver import (
    SAGADouble as _SAGADouble,
    AtomicSAGADouble as _ASAGADouble,
    AtomicSAGADoubleAtomicIterate as _ASAGADoubleA,
    SAGADoubleAtomicIterate as _SAGADoubleA,
    AtomicSAGARelax as _ASAGADoubleRelax,
    SVRGDouble as _SVRGDouble,
    SVRGDoubleAtomicIterate as _SVRGDoubleA,
    SDCADouble as _SDCADouble,
    AtomicSDCADouble as _ASDCADouble,
)

import os

from tick.solver.sdca import AtomicSDCA

data_dir = os.path.join("/".join(__file__.split('/')[:-1]),
                        'saved_solvers')

def find_next_filename(file_prefix):
    import os
    os.makedirs(data_dir, exist_ok=True)

    i = 0
    filename = os.path.join(data_dir, "{}_{}.pkl".format(file_prefix, i))
    while os.path.exists(filename):
        i += 1
        filename = os.path.join(data_dir, "{}_{}.pkl".format(file_prefix, i))

    return filename


def find_last_filename(file_prefix):
    import re
    import os

    file_names = []
    file_regex = "{}_\d+.pkl".format(file_prefix)
    for file_name in os.listdir(data_dir):
        if re.match(file_regex, file_name):
            file_names += [os.path.join(data_dir, file_name)]
    file_names.sort()
    return file_names[-1]


def serialize_history(solver_list, solver_labels, solver_names,
                      n_threads_used, file_prefix, specification):
    import pickle
    histories = [solver.history for solver in solver_list]
    # delete space consumming iterates
    for history in histories:
        if 'x' in history.values.keys():
            del history.values['x']

    file_name = find_next_filename(file_prefix)
    with open(file_name, 'wb') as output_file:
        pickle.dump(dict(histories=histories, solver_labels=solver_labels,
                         solver_names=solver_names,
                         n_threads_used=n_threads_used,
                         specification=specification),
                    output_file)


def load_history(file_prefix):
    import pickle
    file_name = find_last_filename(file_prefix)

    with open(file_name, 'rb') as input_file:
        infos = pickle.load(input_file)

    histories = infos['histories']
    solver_labels = infos['solver_labels']
    solver_names = infos['solver_names']
    n_threads_used = infos['n_threads_used']

    return histories, solver_labels, solver_names, n_threads_used


def compute_l_l2sq(features, labels):
    # return 1. / np.sqrt(len(labels))
    n = len(labels)
    norms = np.power(scipy.sparse.linalg.norm(features, axis=1), 2)
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


def create_solver(solver_class, model, prox, **kwargs):
    if solver_class in [_SAGADouble, _ASAGADouble, _ASAGADoubleA,
                        _ASAGADoubleRelax, _SAGADoubleA,
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
        solver = SDCA(prox.strength, **kwargs)

    elif solver_class == _ASDCADouble:
        solver = AtomicSDCA(prox.strength, **kwargs)
    else:
        raise ValueError('Unknown solver_class {}'.format(solver_class))

    if solver_class in [_SAGADoubleA, _ASAGADoubleA, _SVRGDoubleA]:
        solver.set_model(model.to_atomic()).set_prox(prox.to_atomic())
    else:
        if solver_class in [_SDCADouble, _ASDCADouble]:
            solver.set_model(model).set_prox(ProxZero())
        else:
            solver.set_model(model).set_prox(prox)

    return solver


def train_model(file_prefix, features, labels, specification):
    seed = 4320932
    penalty_strength = compute_l_l2sq(features, labels)  # 1e-5
    print('penalty_strength', penalty_strength)

    model = ModelLogReg(fit_intercept=False)
    model.fit(features, labels)
    prox = ProxL1(penalty_strength)

    test_n_threads = [1, 2, 4]

    solver_list = []
    solver_labels = []
    solver_names = []
    n_threads_used = []

    solver_kwargs = specification['solver_kwargs']
    solver_kwargs.setdefault('max_iter', 10)

    for solver_class in classes:  # [SVRG, SAGA]):
        solver_name = class_names[solver_class]
        print(solver_name)

        for n_threads in test_n_threads:
            print(n_threads)
            solver = create_solver(
                solver_class, model, prox,
                seed=seed, verbose=False, n_threads=n_threads, tol=0,
                **solver_kwargs)

            solver.solve()

            solver_list += [solver]
            solver_labels += ['{} {}'.format(solver_name, n_threads)]
            solver_names += [solver_name]
            n_threads_used += [n_threads]

    serialize_history(solver_list, solver_labels, solver_names, n_threads_used,
                      file_prefix, specification)


def plot_histories(file_prefix, ax, specification):
    histories, solver_labels, solver_names, n_threads_used = \
        load_history(file_prefix)

    threads_ls = {1: '-', 2: '--', 4: ':', 8: '-'}

    plot_history(histories, dist_min=True, log_scale=True,
                 labels=solver_labels, ax=ax, x='time')

    ordered_set_of_solver_names = list(
        OrderedDict.fromkeys(solver_names).keys())
    for line, solver_name, n_threads in zip(ax.lines, solver_names,
                                            n_threads_used):
        color_index = ordered_set_of_solver_names.index(solver_name)
        line.set_color('C{}'.format(color_index))
        line.set_linestyle(threads_ls[n_threads])

    ax.set_ylabel('log distance to optimal objective', fontsize=14)

    fig.tight_layout()
    ax.legend()
    ax.set_ylim([1e-10, None])
    ax.set_title(' '.join(file_prefix.split('_')))


specifications = {
    'synthetic': {
        'dataset_kwargs': {
            'n_samples': 20000,
            'sparsity': 1e-2,
            'nnz': 50,
        },
        'solver_kwargs': {
            'max_iter': 20,
            'record_every': 2,
        }
    },
    'rcv1': {
        'solver_kwargs': {
            'max_iter': 100,
            'record_every': 3,
        }
    }
}


class_names = {
    _SAGADouble: 'Wild',
    _SAGADoubleA: 'Atomic $w$',
    _ASAGADouble: 'Atomic $\\alpha$',
    _ASAGADoubleA: 'Atomic $w$ and $\\alpha$',
    _ASAGADoubleRelax: 'Atomic $\\alpha$ relax',
    _SVRGDouble: 'Wild',
    _SVRGDoubleA: 'Atomic $w$',
    _SDCADouble: 'Wild',
    _ASDCADouble: 'PassCode',
}

class_series = {
    'SVRG': [_SVRGDouble, _SVRGDoubleA],
    'SAGA': [_SAGADouble, _SAGADoubleA, _ASAGADouble, _ASAGADoubleA],
    'SDCA': [_ASDCADouble, _SDCADouble]
}

series = 'SVRG'

if __name__ == '__main__':

    for series in ['SVRG', 'SAGA', 'SDCA']:
        classes = class_series[series]

        dataset_name = 'synthetic'

        specification = specifications[dataset_name]

        file_prefix = '{}_{}'.format(series, dataset_name)

        train = True
        if train:
            features, labels = load_dataset(dataset_name,
                                            specification['dataset_kwargs'])
            print(features.shape, features.nnz / np.prod(features.shape))

            train_model(file_prefix, features, labels, specification)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        plot_histories(file_prefix, ax, specification)
        # plt.xlim([None, 3])
        # plt.ylim([1e-8, None])
        # plt.show()

        os.makedirs('figures', exist_ok=True)
        figure_path = os.path.join("/".join(__file__.split('/')[:-1]),
                                   'figures',
                                   '{}.png'.format(file_prefix))
        plt.savefig(figure_path, dpi=300)

