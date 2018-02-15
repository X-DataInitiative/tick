

from tick.linear_model import SimuLogReg
from tick.simulation import weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np
from tick.online import OnlineForestClassifier
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from time import time


np.set_printoptions(precision=2)

# w0 = weights_sparse_gauss(n_features, nnz=2)
# X, y = SimuLogReg(w0, -1., n_samples=n_samples, seed=seed).simulate()
# y = (y + 1) / 2


def plot_decisions_regression(clfs, datasets, names):
    i = 1
    h = .02
    fig = plt.figure(figsize=(4 * (len(clfs) + 1), 4 * len(datasets)))
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # just plot the dataset first
        cm = plt.cm.RdBu
        ax = plt.subplot(len(datasets), len(clfs) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
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
            clf.fit(X_train, y_train)
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
            i += 1

    plt.tight_layout()
    # plt.show()


def plot_decision_classification(classifiers, datasets, names):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    h = .02
    fig = plt.figure(figsize=(2 * (n_classifiers + 1), 2 * n_datasets))
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
        ax = plt.subplot(n_datasets, n_classifiers + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, cmap=cm)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, s=10,
                   alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
            if hasattr(clf, 'clear'):
                clf.clear()
            if hasattr(clf, 'partial_fit'):
                clf.partial_fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train)
            Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]

            score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
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
    # plt.show()


path = '/Users/stephane.gaiffas/Downloads/'

n_samples = 500
n_features = 2
n_classes = 2

#
# of = OnlineForestClassifier(n_classes=2, n_trees=n_trees, step=30., n_passes=1,
#                             seed=123, use_aggregation=True)
#
# of.fit(X, y)

# print("n_nodes:", of.n_nodes())
# print("n_leaves:", of.n_leaves())
# print(of.predict_proba(X))
# print("step: ", of._forest.step())

# of = OnlineForestClassifier(n_classes=2, n_trees=n_trees, step=1.,
#                             seed=123, use_aggregation=True, n_passes=1)
#
# of.fit(X, y)

# print("n_nodes:", of.n_nodes())
# print("n_leaves:", of.n_leaves())
# print(of.predict_proba(X))
# print("step: ", of._forest.step())


# exit(0)

X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_redundant=0, n_informative=2, random_state=1,
                           n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)


clf = OnlineForestClassifier(n_classes=n_classes, n_trees=50, seed=123,
                             step=1., use_aggregation=True)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

clf.partial_fit(X_train, y_train)

# clf.predict(X_test)

# exit(0)

# clf.print()

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    make_moons(n_samples=n_samples, noise=0.3, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    linearly_separable
]

n_trees = 10

of = OnlineForestClassifier(n_classes=2, n_trees=n_trees, step=30., n_passes=1,
                            seed=123, use_aggregation=True)
seed = 123


params = [
    {'use_aggregation': True, 'n_trees': 50, 'subsampling': 1., 'n_passes': 1, 'dirichlet': 0.1},
    {'use_aggregation': True, 'n_trees': 50, 'subsampling': 1., 'n_passes': 1, 'dirichlet': 0.5},
    {'use_aggregation': True, 'n_trees': 50, 'subsampling': 1., 'n_passes': 1, 'dirichlet': 2},
    # {'use_aggregation': True, 'n_trees': 50, 'subsampling': 1, 'n_passes': 1},
    # {'use_aggregation': True, 'n_trees': 50, 'subsampling': 0.2, 'n_passes': 5},
    # {'use_aggregation': True, 'n_trees': 50, 'subsampling': 0.1, 'n_passes': 10},
    # {'use_aggregation': True, 'n_trees': 1, 'subsampling': 1, 'n_passes': 1},
    # {'use_aggregation': True, 'n_trees': 1, 'subsampling': 0.1, 'n_passes': 10},
    #
    # {'use_aggregation': True, 'n_trees': 5, 'subsampling': 0.2, 'n_passes': 1},
    # {'use_aggregation': True, 'n_trees': 5, 'subsampling': 0.2, 'n_passes': 20},
    # {'use_aggregation': True, 'n_trees': 50, 'subsampling': 0.1, 'n_passes': 1},
    # {'use_aggregation': True, 'n_trees': 50, 'subsampling': 0.1, 'n_passes': 20},
    # {'use_aggregation': False, 'n_trees': 1, 'subsampling': 1, 'n_passes': 1},
    # {'use_aggregation': False, 'n_trees': 1, 'subsampling': 1, 'n_passes': 20},
    # {'use_aggregation': False, 'n_trees': 5, 'subsampling': 0.2, 'n_passes': 1},
    # {'use_aggregation': False, 'n_trees': 5, 'subsampling': 0.2, 'n_passes': 20},
    # {'use_aggregation': False, 'n_trees': 50, 'subsampling': 0.1, 'n_passes': 1},
    # {'use_aggregation': False, 'n_trees': 50, 'subsampling': 0.1, 'n_passes': 20},
]


def toto(kkk):
    return "OF(T: " \
        + str(kkk['n_trees']) + ", S: " + str(kkk['subsampling']) \
        + ', P: ' + str(kkk['n_passes']) + ', di: ' + str(kkk['dirichlet']) \
           + ")"
    # return "OF(A: " + str(kkk['use_aggregation']) + ", T: " \
    #     + str(kkk['n_trees']) + ", S: " + str(kkk['subsampling']) \
    #     + ', P: ' + str(kkk['n_passes']) + ")"


names = list(toto(kw) for kw in params) + ["KNN", "ET", "BRF"]

classifiers = list(
    OnlineForestClassifier(n_classes=n_classes, seed=123, step=1., **kw)
    for kw in params
)

classifiers += [
    KNeighborsClassifier(n_neighbors=5),
    ExtraTreesClassifier(n_estimators=n_trees),
    RandomForestClassifier(n_estimators=n_trees)
]

# names = [
#     "OF(agg, n_passes=1)",
#     "OF(agg, n_passes=5)",
#     "OF(agg, n_passes=10)",
#     "OF(no agg., n_passes=1)",
#     "KNN (k=5)",
#     "ET",
#     "BRF"
# ]

plot_decision_classification(classifiers, datasets, names)

plt.savefig('decisions.pdf')

plt.show()
