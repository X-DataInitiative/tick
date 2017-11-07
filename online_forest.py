from tick.simulation import SimuLinReg, weights_sparse_gauss
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

n_samples = 2000
n_features = 2

w0 = weights_sparse_gauss(n_features, nnz=2)
X, y = SimuLinReg(w0, -1., n_samples=n_samples).simulate()
X_train, X_test, y_train, y_test = train_test_split(X, y)


def plot_decision_regions(clf, X, y, n_iter=None):
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

    clf.set_data(X, y)
    clf.fit(n_iter)

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ct = plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cm)
    plt.colorbar(ct)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.xlabel('x1', fontsize=16)
    plt.ylabel('x2', fontsize=16)
    plt.legend(loc='upper left')
    plt.tight_layout()


from tick.inference import OnlineForestRegressor

clf = OnlineForestRegressor(n_trees=100, min_samples_split=1)

# clf.set_data(X, y)
# clf.fit(20)

# clf.print()

# exit(0)
# forest = OnlineForestRegressor(n_trees=100, min_samples_split=50)

plot_decision_regions(clf, X, y, n_samples)
plt.show()

# plt.savefig('/Users/stephane.gaiffas/Downloads/online-forest.pdf')
