from tick.simulation import SimuLogReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np
from tick.inference import OnlineForestClassifier
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tick.simulation import weights_sparse_exp, SimuLogReg

np.set_printoptions(precision=2)

# w0 = weights_sparse_gauss(n_features, nnz=2)
# X, y = SimuLogReg(w0, -1., n_samples=n_samples, seed=seed).simulate()


n_samples = 10000
n_features = 50
n_classes = 2


w0 = weights_sparse_exp(n_features, nnz=10)
X, y = SimuLogReg(weights=w0, intercept=None, n_samples=n_samples).simulate()
y = (y + 1) / 2

path = '/Users/stephane.gaiffas/Downloads/'


clf = OnlineForestClassifier(n_classes=n_classes, n_trees=50, seed=123,
                             step=1., use_aggregation=True)

clf.fit(X, y)


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


# X_train, X_test, y_train, y_test = \
#     train_test_split(X, y, test_size=.4, random_state=42)
#
# clf.fit(X_train, y_train)

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

# plt.savefig('decisions.pdf')

plt.show()
