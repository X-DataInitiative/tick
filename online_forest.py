from tick.simulation import SimuLinReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np
from tick.inference import OnlineForestRegressor

import matplotlib.pyplot as plt

n_samples = 50
n_features = 2
seed = 123

np.set_printoptions(precision=2)


w0 = weights_sparse_gauss(n_features, nnz=2)
X, y = SimuLinReg(w0, -1., n_samples=n_samples, seed=seed).simulate()

X_train, X_test, y_train, y_test = train_test_split(X, y)


def plot_decision_regions(clf, X, y, n_iter=None, use_aggregation=None):
    from matplotlib.colors import ListedColormap

    cm = plt.cm.RdBu
    cmap = ListedColormap(['red', 'white', 'blue'])
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(1, 1, 1)
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))

    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap=cm)

    if use_aggregation is None:
        Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    else:
        Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T, use_aggregation)
    Z = Z.reshape(xx1.shape)
    ct = plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cm)
    plt.colorbar(ct)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.title('Online random forest, aggregation= ' + str(use_aggregation))
    plt.legend(loc='upper left')
    plt.tight_layout()


# clf = OnlineForestRegressor(n_trees=1, seed=123)
# print(clf.predict(X))

# plot_decision_regions(clf, X, y, use_aggregation=False)

path = '/Users/stephane.gaiffas/Downloads/'

import os

# plt.savefig(os.path.join(path, 'online1.pdf'))

clf = OnlineForestRegressor(n_trees=10, seed=123)
clf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

et = ExtraTreesRegressor()
et.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y1 = clf.predict(X_test, use_aggregation=False)
y2 = clf.predict(X_test, use_aggregation=True)
y3 = rf.predict(X_test)
y4 = et.predict(X_test)

print(y_test)
print(y1)
print(y2)

print("err 1-NN: ", np.linalg.norm(y_test - y1))
print("err agg: ", np.linalg.norm(y_test - y2))
print("err breiman: ", np.linalg.norm(y_test - y3))
print("err et: ", np.linalg.norm(y_test - y4))


plot_decision_regions(rf, X, y)

plot_decision_regions(clf, X, y, use_aggregation=False)

plot_decision_regions(clf, X, y, use_aggregation=True)
plt.show()

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
