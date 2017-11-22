
import os
import pandas as pd
import pickle as pkl

from tick.inference import OnlineForestRegressor, OnlineForestClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    # "OF (agg, step=1.)",
    # "OF(agg, step=100.)",
    "OF(no agg.)",
    "KNN (k=5)",
    "ET",
    "BRF"
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

    # online_forest.set_probabilities(probabilities)
    classifiers = [
        # OnlineForestClassifier(n_trees=n_trees, seed=123, step=1.,
        #                        use_aggregation=True),
        # OnlineForestClassifier(n_trees=n_trees, seed=123, step=100.,
        #                        use_aggregation=True),
        OnlineForestClassifier(n_trees=n_trees, seed=123, step=1.,
                               use_aggregation=False),
        KNeighborsClassifier(n_neighbors=5),
        ExtraTreesClassifier(n_estimators=n_trees),
        RandomForestClassifier(n_estimators=n_trees)
    ]

    for clf, name in zip(classifiers, names):
        clf.fit(X_train, y_train)
        # print('Accuracy of', name, ': ', '%.2f' % clf.score(X_test, y_test))
