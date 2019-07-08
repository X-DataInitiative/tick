# License: BSD 3 clause

import os
import pprint
import time
import warnings
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKlearn
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

from tick.dataset import fetch_tick_dataset
from tick.dataset.url_dataset import fetch_url_dataset
from tick.inference import LogisticRegression as LogisticRegressionTick


def find_best_C(learner, X, y):
    n_samples = len(y)
    scores = {}
    next_test_C = np.logspace(4, 6, 4)

    for _ in range(5):

        next_test_C = np.array(next_test_C)
        if isinstance(learner, LogisticRegressionSKlearn):
            next_C_param = next_test_C / n_samples
        else:
            next_C_param = next_test_C

        print('triying', next_test_C)
        parameters = {'C': next_C_param}

        learner_cv = GridSearchCV(learner, parameters, scoring='roc_auc',
                                  return_train_score=True, cv=5, n_jobs=1)

        with warnings.catch_warnings(record=True) as w_list:
            learner_cv.fit(X, y)

        for warning in w_list:
            if warning.category == DeprecationWarning:
                # print('skipped', warning)
                pass
            else:
                warnings.warn(warning.message, warning.category)

        new_scores = learner_cv.cv_results_['mean_test_score']

        used_parameters = learner_cv.cv_results_['params']

        for params, score in zip(used_parameters, new_scores):
            if isinstance(learner, LogisticRegressionSKlearn):
                params['C'] *= n_samples

            if params['C'] in scores:
                print('WAS ALREADY TRIED', params['C'], scores)
            scores[params['C']] = score

        print(scores)

        best_C = max(scores.items(), key=lambda x: x[1])[0]
        tested_C = list(scores.keys())
        tested_C.sort()

        # all scores are equal
        if min(scores.values()) == max(scores.values()):
            next_test_C = [tested_C[0] * 0.1, tested_C[-1] * 10]
            continue

        best_C_index = tested_C.index(best_C)
        if best_C_index == 0:
            left_C = best_C * 0.01
            right_C = best_C * 0.1
        elif best_C_index == len(tested_C) - 1:
            right_C = best_C * 100
            left_C = best_C * 10
        else:
            left_C = np.sqrt(best_C * tested_C[best_C_index - 1])
            right_C = np.sqrt(best_C * tested_C[best_C_index + 1])

        next_test_C = [left_C, right_C]
        # print(next_test_C)

    best_C = max(scores.items(), key=lambda x: x[1])[0]
    return best_C, scores


def run_best_C(learner, X, y, tick_learner, test_size):
    np.random.seed(29389328)
    n_samples = len(y)
    shuffled_index = np.random.permutation(np.arange(n_samples))
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]

    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    start_time = time.time()
    learner.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    fpr, tpr, _ = roc_curve(y_test, learner.predict_proba(X_test)[:, 1])
    auc_value = auc(fpr, tpr)

    if isinstance(learner, LogisticRegressionSKlearn):
        #coeffs = np.hstack((learner.coef_[0], learner.intercept_))
        coeffs = learner.coef_[0]
    else:
        coeffs = learner._solver_obj.solution

    tick_learner._model_obj.fit(X_train, y_train)
    train_objective = tick_learner._model_obj.loss(coeffs)
    train_objective += tick_learner._prox_obj.value(coeffs)

    return learner, elapsed_time, auc_value, train_objective

def plot_scores(score_dict, ax, lib, c, auc_value):
    tested_C = list(score_dict.keys())
    tested_C.sort()
    scores = np.array(list(map(lambda c: score_dict[c], tested_C)))
    scores = max(scores) - scores
    scores += 3 * min(scores[scores != 0])
    ax.plot(tested_C, scores, marker='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('{}, C={:.5g}, AUC={:.4f}'.format(lib, c, auc_value))


def write_to_file(file_path, text):
    directory = os.path.dirname(file_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    with open(file_path, 'a') as f:
        f.write(text)


def load_dataset():
    n_url_days = 1
    dataset_file_name = 'url_d{}'.format(n_url_days)

    X, y = fetch_url_dataset(n_url_days)
    print('SHAPES', X.shape, y.shape)

    # dataset_file_name = 'breast'
    # X, y = datasets.load_breast_cancer(return_X_y=True)

    # dataset_file_name = 'adult'
    # X, y = fetch_tick_dataset('binary/adult/adult.trn.bz2')
    return dataset_file_name, X, y


def find_best_params():
    dataset_file_name, X, y = load_dataset()

    file_name = 'results/{}_{}.txt'.format(dataset_file_name, max_iter)

    sckit_learner = LogisticRegressionSKlearn(solver='saga', max_iter=max_iter,
                                              penalty='l1', tol=tol,
                                              fit_intercept=False)
    tick_learner = LogisticRegressionTick(solver='saga', max_iter=max_iter,
                                          penalty='l1', tol=tol,
                                          fit_intercept=False)

    best_C_scikit, scikit_score = find_best_C(sckit_learner, X, y)
    write_to_file(file_name,
                  'Scikit scores \n{}\n'.format(pprint.pformat(scikit_score)))
    best_C_tick, tick_score = find_best_C(tick_learner, X, y)
    write_to_file(file_name,
                  'Tick scores \n{}\n'.format(pprint.pformat(tick_score)))

    write_to_file(file_name, '\nScikit, best C={}, AUC={}\n'
                  .format(best_C_scikit, scikit_score[best_C_scikit]))
    write_to_file(file_name, 'Tick, best C={}, AUC={}\n'
                  .format(best_C_tick, tick_score[best_C_tick]))

    fig, ax_list = plt.subplots(1, 2, sharey=True, figsize=(10, 3))
    plot_scores(scikit_score, ax_list[0], 'scikit', best_C_scikit,
                scikit_score[best_C_scikit])
    plot_scores(tick_score, ax_list[1], 'tick', best_C_tick,
                tick_score[best_C_tick])
    plt.savefig(file_name.replace('.txt', '.pdf'))


def plot_best_params(C):
    dataset_file_name, X, y = load_dataset()
    print(X.shape)

    test_size = int(0.2 * len(y))
    train_size = len(y) - test_size

    tick_learner = LogisticRegressionTick(C=C, penalty='l1',
                                          fit_intercept=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    times = {}
    auc_values = {}
    objectives = {}

    libs = ['tick', 'scikit']
    for lib in libs:
        times[lib] = []
        auc_values[lib] = []
        objectives[lib] = []

        for max_iter in [10 * i for i in range(1, 11)]:# + [200, 300]:

            if lib == 'scikit':
                learner = LogisticRegressionSKlearn(solver='saga',
                                                    max_iter=max_iter,
                                                    tol=tol,
                                                    C=C / train_size,
                                                    penalty='l1',
                                                    fit_intercept=False)
            else:
                learner = LogisticRegressionTick(solver='saga',
                                                 max_iter=max_iter,
                                                 tol=tol,
                                                 record_every=10000,
                                                 print_every=10000,
                                                 verbose=True,
                                                 C=C, penalty='l1',
                                                 random_state=10392,
                                                 fit_intercept=False)

            _, elapsed_time, auc_value, train_objective = run_best_C(
                learner, X, y, tick_learner, test_size)

            times[lib] += [elapsed_time]
            auc_values[lib] += [auc_value]
            objectives[lib] += [train_objective]

        print(objectives)

    min_objectives = min(chain.from_iterable([objectives[lib] for lib in libs]))
    for lib in libs:
        axes[0].plot(times[lib], auc_values[lib], marker='x', label=lib)

        lib_objectives = np.array(objectives[lib]) - min_objectives
        lib_objectives += min(lib_objectives[lib_objectives != 0]) / 2
        axes[1].plot(times[lib], lib_objectives,
                     marker='x', label=lib)

    axes[1].set_yscale('log')

    axes[0].set_xlabel('time')
    axes[0].set_title('AUC')
    axes[1].set_xlabel('time')
    axes[1].set_title('distance to optimal objective')
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    plt.show()



max_iter = 100
tol = 1e-16
# find_best_params()
plot_best_params(1e10)

# dataset_file_name, X, y = load_dataset()
# print(X.shape)
#
# test_size = int(0.2 * len(y))
# train_size = len(y) - test_size
#
# fit_intercept = False
# tick_learner = LogisticRegressionTick(C=1e5, penalty='l1', fit_intercept=fit_intercept)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
# times = {}
# auc_values = {}
# objectives = {}
#
# libs = ['scikit', 'tick']
# for lib in libs:
#     times[lib] = []
#     auc_values[lib] = []
#     objectives[lib] = []
#
#     for max_iter in [10 * i for i in range(1, 11)] + [200, 300]:
#
#         if lib == 'scikit':
#             learner = LogisticRegressionSKlearn(solver='saga',
#                                                 max_iter=max_iter,
#                                                 tol=tol,
#                                                 C=1e5 / train_size,
#                                                 penalty='l1',
#                                                 verbose=True,
#                                                 fit_intercept=fit_intercept)
#         else:
#             learner = LogisticRegressionTick(solver='saga',
#                                              max_iter=max_iter,
#                                              tol=tol,
#                                              record_every=10000,
#                                              print_every=10000,
#                                              verbose=False,
#                                              C=1e5, penalty='l1',
#                                              random_state=10392,
#                                              fit_intercept=fit_intercept)
#
#             save_learner = learner
#
#         _, elapsed_time, auc_value, train_objective = run_best_C(
#             learner, X, y, tick_learner, test_size)
#
#         times[lib] += [elapsed_time]
#         auc_values[lib] += [auc_value]
#         objectives[lib] += [train_objective]
#
#     print(objectives)
#
# min_objectives = min(chain.from_iterable([objectives[lib] for lib in libs]))
# for lib in libs:
#     axes[0].plot(times[lib], auc_values[lib], marker='x', label=lib)
#
#     lib_objectives = np.array(objectives[lib]) - min_objectives
#     lib_objectives += min(lib_objectives[lib_objectives != 0]) / 2
#     axes[1].plot(times[lib], lib_objectives,
#                  marker='x', label=lib)
#
# axes[1].set_yscale('log')
#
# axes[0].set_xlabel('time')
# axes[0].set_title('AUC')
# axes[1].set_xlabel('time')
# axes[1].set_title('distance to optimal objective')
# axes[0].legend()
# axes[1].legend()
#
# fig.tight_layout()
# plt.show()
