from time import time
from sklearn.model_selection import train_test_split
import sys

sys.path.append('/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/')

from datasets import readers as all_readers
from skgarden import MondrianForestClassifier
from tick.online import OnlineForestClassifier

path = '/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/datasets/'

readers = all_readers

X, y, dataset_name = all_readers[1](path)

print("X.shape=", X.shape)


X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.3, random_state=42)


clf = MondrianForestClassifier(n_estimators=50, min_samples_split=1)

t1 = time()
clf.partial_fit(X_train, y_train)
t2 = time()
print('MF:', t2 - t1, 'Acc:', clf.score(X_test, y_test))

of = OnlineForestClassifier(n_classes=2, n_trees=50)

t1 = time()
of.partial_fit(X_train, y_train)
t2 = time()
print('OF:', t2 - t1, 'Acc:', of.score(X_test, y_test))
