from tick.simulation import SimuLinReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np
from tick.inference import OnlineForestRegressor
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt

from time import time

n_samples = 2000
n_features = 2
seed = 123

np.set_printoptions(precision=2)


w0 = weights_sparse_gauss(n_features, nnz=2)
X, y = SimuLinReg(w0, -1., n_samples=n_samples, seed=seed).simulate()

# load_boston([return_X_y])
# load_diabetes([return_X_y])


def plot_decisions(clfs, datasets, names, use_aggregation=None):
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
            if isinstance(clf, OnlineForestRegressor):
                clf.clear()
            t1 = time()
            clf.fit(X_train, y_train)
            t2 = time()

            mse = np.linalg.norm(y_test - clf.predict(X_test))
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

            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f (%.2f)' % (mse, t2-t1)).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    # plt.show()


# def plot_decision_regions(clfs, X_test, y_test, n_iter=None, use_aggregation=None,
#                           title=None):
#     from matplotlib.colors import ListedColormap
#
#     cm = plt.cm.RdBu
#     cmap = ListedColormap(['red', 'white', 'blue'])
#     fig = plt.figure(figsize=(8, 5))
#
#     ax = plt.subplot(1, 1, 1)
#     # plot the decision surface
#     x1_min, x1_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
#     x2_min, x2_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
#
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
#                            np.arange(x2_min, x2_max, 0.02))
#
#     plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10, cmap=cm)
#
#     if use_aggregation is None:
#         Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     else:
#         Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T, use_aggregation)
#     Z = Z.reshape(xx1.shape)
#     ct = plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cm)
#     plt.colorbar(ct)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     plt.xlabel('x1', fontsize=16)
#     plt.ylabel('x2', fontsize=16)
#     if title is not None:
#         plt.title(title)
#     plt.legend(loc='upper left')
#     plt.tight_layout()


# clf = OnlineForestRegressor(n_trees=1, seed=123)
# print(clf.predict(X))

# plot_decision_regions(clf, X, y, use_aggregation=False)

path = '/Users/stephane.gaiffas/Downloads/'

import os

# plt.savefig(os.path.join(path, 'online1.pdf'))

n_trees = 5


from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

X_moons, _ = make_moons(n_samples=500, noise=0.3, random_state=0)
X_moons = StandardScaler().fit_transform(X_moons)
y_moons = X_moons.dot(w0) - 1


# datasets = [make_moons(noise=0.3, random_state=0),
#             make_circles(noise=0.2, factor=0.5, random_state=1),
#             linearly_separable
#             ]
datasets = [
    (X, y),
    (X_moons, y_moons)
]

clfs = [
    OnlineForestRegressor(n_trees=n_trees, seed=123, step=0.5),
    ExtraTreesRegressor(n_estimators=n_trees),
    RandomForestRegressor(n_estimators=n_trees)
]

names = [
    "Online forest",
    "Extra trees",
    "Breiman RF"
]

# X_train, X_test, y_train, y_test = train_test_split(X, y)


# for clf, name in zip(clfs, names):
#     clf.fit(X_train, y_train)
#     pred = clf.predict(X_test)
#     print(name, "mse: ", np.linalg.norm(y_test - pred))


plot_decisions(clfs, datasets, names)
plt.show()

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
