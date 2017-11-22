
import os
import pandas as pd
import pickle as pkl

from tick.inference import OnlineForestRegressor, OnlineForestClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier

import matplotlib.pyplot as plt

path = '/Users/stephane.gaiffas/Dropbox/jaouad/online-forests/datasets/'

filenames = [
    'dna.p',
    'letter.p',
    'satimage.p',
    'usps.p'
]

n_classess = [3, 25, 5, 9]

n_trees = 10

names = [
    "Online forest",
    "Extra trees",
    "Breiman RF"
]

for filename, n_classes in zip(filenames, n_classess):
    print(filename)
    with open(os.path.join(path, filename), 'rb') as f:
        data = pkl.load(f)
        X_train = data['x_train']
        X_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']

    # triche = RandomForestClassifier(n_estimators=n_trees)
    # triche.fit(X_train, y_train)
    # probabilities = triche.feature_importances_ / triche.feature_importances_.sum()
    #
    # plt.stem(probabilities)
    # plt.title('Features importance for ' + filename, fontsize=18)
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # # plt.show()
    # plt.savefig(filename + '.pdf')

    online_forest = OnlineForestClassifier(n_trees=n_trees, n_classes=n_classes,
                                           seed=123, step=1.)
    # online_forest.set_probabilities(probabilities)
    classifiers = [
        online_forest,
        ExtraTreesClassifier(n_estimators=n_trees),
        RandomForestClassifier(n_estimators=n_trees)
    ]

    for clf, name in zip(classifiers, names):
        clf.fit(X_train, y_train)
        print('Accuracy of', name, ': ', '%.2f' % clf.score(X_test, y_test))
