
from matplotlib.colors import ListedColormap
import os
from sklearn.model_selection import train_test_split
import numpy as np
from tick.online import OnlineForestClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

n_samples = 1000
n_features = 2
seed = 123

X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=0)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.5, random_state=42)

h = .1
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.zeros(xx.shape)

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


n_trees = 10

clf = OnlineForestClassifier(n_classes=2, n_trees=n_trees, seed=123, step=1.,
                             use_aggregation=True)

save_iterations = [5, 10, 30, 50, 100, 300]
n_plots = len(save_iterations)
n_fig = 0

fig = plt.figure(figsize=(3 * n_plots, 3.2))


for i in range(X_train.shape[0]):

    if hasattr(clf, 'partial_fit'):
        clf.partial_fit(X_train[i, :].reshape(1, 2), np.array([y_train[i]]))
    else:
        if i:
            clf.fit(X_train[:i, :], y_train[:i])

    if i in save_iterations:
        n_fig += 1
        Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]
        Z = Z.reshape(xx.shape)
        ax = plt.subplot(1, n_plots, n_fig)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:i, 0], X_train[:i, 1], c=y_train[:i], s=25, cmap=cm)
        score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        ax.set_title('t = %d' % i, fontsize=20)

        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=19, horizontalalignment='right')

        # print('iteration= %d' % i)


plt.tight_layout()



path = '/Users/stephane.gaiffas/Code/tick'
plt.savefig(os.path.join(path, 'of_iterations.pdf'))

# plt.show()

# def animate(i):
#     clf.partial_fit(X_train[i, :].reshape(1, 2), np.array([y_train[i]]))
#     Z = clf.predict_proba(np.array([xx.ravel(), yy.ravel()]).T)[:, 1]
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, cmap=cm, alpha=.5)
#     ax.scatter(X_train[:i, 0], X_train[:i, 1], c=y_train[:i], s=25, cmap=cm)
#     score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
#     fig.suptitle('test auc: %.2f' % score, fontsize=18)
#     return ax
#
# # Interval in seconds
# interval = 10
# ani = animation.FuncAnimation(fig, animate, 200, interval=interval)
#
# plt.show()
