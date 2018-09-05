
import os
import pandas as pd
import pickle as pkl

from tick.online import OnlineForestRegressor, OnlineForestClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


# TODO: options for types of sampling of the features
# TODO: online construction of the feature_importances
# TODO: python script that tries all combinations

# TODO: what if we feed several times the same dataset
# TODO: show that the classifier is insensitive to the time of arrival of the points
# TODO: V-fold instead of train and test ?
# TODO: Set features importance with default to none
# TODO: implement a subsample strategy : only one tree is updated with the given sample
# TODO: tree aggregation
# TODO: different "types" of trees: no aggregation, aggregation and different temperatures

# TODO: unittest for attributes
# TODO: unittest for wrong n_features in fit and predict and wrong labels in training

# TODO: tryout multiple passes
# TODO: really make seed work with inline forest


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
    "OF (agg, step=1.)",
    "OF(agg, step=100.)",
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

    classifiers = [
        OnlineForestClassifier(n_trees=n_trees, seed=123, step=1.,
                               use_aggregation=True, n_classes=n_classes),
        OnlineForestClassifier(n_trees=n_trees, seed=123, step=100.,
                               n_classes=n_classes, use_aggregation=True),
        OnlineForestClassifier(n_trees=n_trees, seed=123, step=1.,
                               use_aggregation=False, n_classes=n_classes),
        KNeighborsClassifier(n_neighbors=5),
        ExtraTreesClassifier(n_estimators=n_trees),
        RandomForestClassifier(n_estimators=n_trees)
    ]

    triche = RandomForestClassifier(n_estimators=n_trees)
    triche.fit(X_train, y_train)
    feature_importances = triche.feature_importances_ / triche.feature_importances_.sum()
    #
    # plt.stem(probabilities)
    # plt.title('Features importance for ' + filename, fontsize=18)
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # # plt.show()
    # plt.savefig(filename + '.pdf')

    # online_forest.set_probabilities(probabilities)

    # forest1 =

    for clf, name in zip(classifiers, names):
        if hasattr(clf, 'clear'):
            clf.clear()
            clf.set_feature_importances(feature_importances)
        # print('Fitting', name)
        clf.fit(X_train, y_train)
        # print('Done.')
        print('Accuracy of', name, ': ', '%.2f' % clf.score(X_test, y_test))
