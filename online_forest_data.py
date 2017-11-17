
import os
import pandas as pd
import pickle as pkl

from tick.inference import OnlineForestRegressor, OnlineForestClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    RandomForestClassifier, ExtraTreesClassifier

path = '/Users/stephane.gaiffas/Dropbox/jaouad/online-forests/datasets/'


# filename = 'dna.p'
# filename = 'letter.p'
# filename = 'satimage.p'
filename = 'usps.p'

with open(os.path.join(path, filename), 'rb') as f:
    data = pkl.load(f)

X_train = data['x_train']
X_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

n_classes = y_train.max()

n_classes = 5

print("n_classes:", n_classes)
n_trees = 10

classifiers = [
    OnlineForestClassifier(n_trees=n_trees, n_classes=n_classes,
                           seed=123, step=1.),
    ExtraTreesClassifier(n_estimators=n_trees),
    RandomForestClassifier(n_estimators=n_trees)
]
names = [
    "Online forest",
    "Extra trees",
    "Breiman RF"
]

for clf, name in zip(classifiers, names):
    clf.fit(X_train, y_train)
    print('Accuracy of', name, ': ', '%.2f' % clf.score(X_test, y_test))
