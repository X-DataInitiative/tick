# License: BSD 3 clause

import os
import pprint
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression as LogisticRegressionSKlearn
from sklearn.model_selection import GridSearchCV

from sklearn import datasets
from tick.dataset.url_dataset import fetch_url_dataset
from tick.inference import LogisticRegression as LogisticRegressionTick


def find_best_C(learner, X, y):
    scores = {}
    next_test_C = np.logspace(-5, 5, 20)

    for _ in range(5):
        print('triying', next_test_C)
        parameters = {'C': next_test_C}

        learner_cv = GridSearchCV(learner, parameters, scoring='roc_auc',
                                  return_train_score=True, cv=5, n_jobs=20)

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
            if params['C'] in scores:
                print('WAS ALREADY TRIED', params['C'], scores)
            scores[params['C']] = score

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


def plot_scores(score_dict, ax, lib, c, auc):
    tested_C = list(score_dict.keys())
    tested_C.sort()
    scores = np.array(list(map(lambda c: score_dict[c], tested_C)))
    scores = max(scores) - scores
    scores += 3 * min(scores[scores != 0])
    ax.plot(tested_C, scores, marker='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('{}, C={:.5g}, AUC={:.4f}'.format(lib, c, auc))


def write_to_file(file_path, text):
    directory = os.path.dirname(file_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

    with open(file_path, 'a') as f:
        f.write(text)


n_url_days = 10
dataset_file_name = 'url_d{}'.format(n_url_days)
X, y = fetch_url_dataset(n_url_days)
print('SHAPES', X.shape, y.shape)

# dataset_file_name = 'breast'
# X, y = datasets.load_breast_cancer(return_X_y=True)

tol = 1e-16
max_iter = 100

file_name = 'results/{}_{}.txt'.format(dataset_file_name, max_iter)

sckit_learner = LogisticRegressionSKlearn(solver='saga', max_iter=max_iter,
                                          penalty='l1', tol=tol)
tick_learner = LogisticRegressionTick(solver='saga', max_iter=max_iter,
                                      penalty='l1', tol=tol)

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
