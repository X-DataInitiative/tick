
import matplotlib.animation as animation
from matplotlib.animation import MovieWriter

from sklearn.model_selection import train_test_split
import numpy as np
from tick.online import OnlineForestClassifier
from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

n_samples = 500
n_features = 2
seed = 123

X, y = make_moons(n_samples=n_samples, noise=0.3, random_state=0)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.5, random_state=42)

h = .1
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.zeros(xx.shape)

cm = plt.cm.RdBu

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(1, 1, 1)


ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

ax.scatter(X_train[:2, 0], X_train[:2, 1], c=np.array([0, 1]), s=25, cmap=cm)

n_trees = 20

clf = OnlineForestClassifier(n_classes=2, n_trees=n_trees, seed=123, step=1.,
                             use_aggregation=False)

# print("clf=", clf)


clf.partial_fit(X, y)

exit(0)


def animate(i):
    clf.partial_fit(X_train[i, :].reshape(1, 2), np.array([y_train[i]]))
    Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.5)
    ax.scatter(X_train[:i, 0], X_train[:i, 1], c=y_train[:i], s=25, cmap=cm)
    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    fig.suptitle('test auc: %.2f' % score, fontsize=18)
    return ax

# Interval in seconds
interval = 10
ani = animation.FuncAnimation(fig, animate, 200, interval=interval)

plt.show()
