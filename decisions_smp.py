from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib.colors import ListedColormap

from tick.linear_model import SimuLogReg
from tick.linear_model import LogisticRegression

from sklearn.datasets import make_moons, make_classification, make_circles
from sklearn.metrics import roc_auc_score

import logging

import matplotlib.pyplot as plt
from tick.plot import stems

from skgarden import MondrianForestClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

np.set_printoptions(precision=2)


def plot_decision_classification(classifiers, datasets):
    n_classifiers = len(classifiers)
    n_datasets = len(datasets)
    h = .2
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
        for name, clf in classifiers:
            ax = plt.subplot(n_datasets, n_classifiers + 1, i)
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


# Simulation of datasets
n_samples = 100
n_samples_half = int(n_samples / 2)
n_features = 2
n_classes = 2
random_state = 42

X = 0.5 * np.random.randn(n_samples, n_features)

X[:n_samples_half] += np.array([1., 2.])
X[n_samples_half:] += np.array([-2., -1.])

y = np.zeros(n_samples)
y[n_samples_half:] = 1
y[:n_samples_half] = -1

# X, y = SimuLogReg(weights=np.array([1., 1.]), intercept=0., features=X).simulate()

# X, y = make_classification(n_samples=n_samples, n_features=n_features,
#                            n_redundant=0, n_informative=2,
#                            random_state=random_state,
#                            n_clusters_per_class=1)
#
# rng = np.random.RandomState(random_state)
# X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)

datasets = [
    make_moons(n_samples=n_samples, noise=0.1, random_state=0),
    make_circles(n_samples=n_samples, noise=0.1, factor=0.5,
                 random_state=random_state),
    linearly_separable
]


classifiers = [
    ('LR', LogisticRegression(random_state=123, penalty='none', tol=0, solver='svrg', smp=False)),
    ('LR SMP', LogisticRegression(random_state=123, penalty='none', tol=0, solver='svrg', smp=True)),
    # ('MF', MondrianForestClassifier(n_estimators=n_trees)),
    # ('RF', RandomForestClassifier(n_estimators=n_trees)),
    # ('ET', ExtraTreesClassifier(n_estimators=n_trees))
]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

plot_decision_classification(classifiers, datasets)

logging.info("Saved the decision functions in 'decision.pdf")

plt.savefig('decisions.pdf')

for _, clf in classifiers:
    print(clf.weights)

