from tick.simulation import SimuLinReg, SimuLogReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np
from tick.inference import OnlineForestRegressor, OnlineForestClassifier
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from time import time

n_samples = 500
n_features = 2
seed = 123

np.set_printoptions(precision=2)


w0 = weights_sparse_gauss(n_features, nnz=2)
X, y = SimuLogReg(w0, -1., n_samples=n_samples, seed=seed).simulate()
y = (y + 1) / 2


# X_train, X_test, y_train, y_test = train_test_split(X, y)


def plot_decisions_regression(clfs, datasets, names, use_aggregation=None):
    i = 1
    h = .02
    fig = plt.figure(figsize=(4 * (len(clfs) + 1), 4 * len(datasets)))
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        # X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(clfs) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        #     plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10, cmap=cm)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=25, cmap=cm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=25,
                   alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, clfs):
            ax = plt.subplot(len(datasets), len(clfs) + 1, i)

            t1 = time()
            clf.fit(X_train, y_train)
            t2 = time()

            # mse = np.linalg.norm(y_test - clf.predict(X_test))
            # score = clf.score(X_test, y_test)

            Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, s=15)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm,
                       s=15, alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)

            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f (%.2f)' % (mse, t2-t1)).lstrip('0'),
            #         size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    # plt.show()


def plot_decision_classification(classifiers, datasets, names):
    h = .02
    fig = plt.figure(figsize=(2 * (len(classifiers) + 1), 2 * len(datasets)))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10,
                   alpha=0.6)

        # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
        #            edgecolors='k')
        # # and testing points
        # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
        #            alpha=0.6,
        #            edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            if hasattr(clf, 'clear'):
                clf.clear()
            clf.fit(X_train, y_train)

            Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]

            score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            # score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to
            # each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]


            # Z = Z[:, 1]
            # print(Z)
            # print(Z.shape)
            # print(xx.shape, xx.shape[0] * xx.shape[1])

            # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, s=15)
            # # and testing points
            # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm,
            #            s=15, alpha=0.6)

            # ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
            #            edgecolors='k')
            # # and testing points
            # ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
            #            edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()


path = '/Users/stephane.gaiffas/Downloads/'

import os

# plt.savefig(os.path.join(path, 'online1.pdf'))

n_trees = 50

X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(n_samples=n_samples, noise=0.3, random_state=0),
            make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# datasets = [
#     (X, y)
# ]

# clfs = [
#     OnlineForestClassifier(n_trees=n_trees, seed=123, step=0.25),
#     ExtraTreesRegressor(n_estimators=n_trees),
#     RandomForestRegressor(n_estimators=n_trees)
# ]

classifiers = [
    OnlineForestClassifier(n_trees=n_trees, seed=123, step=1.),
    ExtraTreesClassifier(n_estimators=n_trees),
    RandomForestClassifier(n_estimators=n_trees)
]
names = [
    "Online forest",
    "Extra trees",
    "Breiman RF"
]


# forest = OnlineForestClassifier(n_trees=n_trees, n_classes=2, seed=123, step=1.)
# print(y)

# forest.fit(X, y)
# forest.predict(X)


plot_decision_classification(classifiers, datasets, names)
plt.show()

# forest = OnlineForestRegressor(n_trees=n_trees, seed=123, step=0.25)
#
# forest.fit(X, y)
#
# forest.predict(X)

# plt.savefig(os.path.join(path, 'decisions.pdf'))


# plot_decision_regions(clf, X, y, use_aggregation=True)

# plt.savefig(os.path.join(path, 'online2.pdf'))

# clf.print()

# plt.show()


# clf.fit(X, y)

# print(y)
# print(clf.predict(X))
# clf.print()


# plot_decision_regions(clf, X, y, n_iter=None, use_aggregation=True)
# plt.show()

# exit(0)
# forest = OnlineForestRegressor(n_trees=100, min_samples_split=50)

# plot_decision_regions(clf, X, y, n_samples)


# plt.savefig('/Users/stephane.gaiffas/Downloads/online-forest.pdf')
