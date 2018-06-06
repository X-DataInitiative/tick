from time import time
from sklearn.model_selection import train_test_split
import sys

sys.path.append('/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/')

from datasets import readers as all_readers
from skgarden import MondrianForestClassifier

from tick.online import OnlineForestClassifier
from sklearn.model_selection import StratifiedKFold

path = '/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/datasets/'

readers = all_readers

X, y, dataset_name = all_readers[1](path)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.3, random_state=42)

#
# clf = MondrianForestClassifier(n_estimators=1, min_samples_split=1)
#
# t1 = time()
# clf.partial_fit(X_train, y_train)
# t2 = time()
# print('MF:', t2 - t1, 'Acc:', clf.score(X_test, y_test))
#
# exit(0)

of = OnlineForestClassifier(n_classes=2, n_trees=10, split_pure=False,
                            min_samples_split=3)
t1 = time()
# of.partial_fit(X_train[:4], y_train[:4])
of.partial_fit(X_train, y_train)
t2 = time()
print('OF:', t2 - t1, 'Acc:', of.score(X_test, y_test))



# nodes = of1.get_nodes(0)

exit(0)

df = of1.get_nodes_df(0)

print(df)
# import json
#
# print(json.dumps(nodes, indent=2, sort_keys=True))
# t1 = time()
# for i in range(10)
# of.partial_fit(X_train[:10], y_train[:10])


exit(0)

of2 = OnlineForestClassifier(n_classes=2, n_trees=1)

n_minibatches = 100
skf = StratifiedKFold(n_splits=n_minibatches, shuffle=True)
indices = [test for (train, test) in skf.split(X, y)]

for idx in indices:
    X_train, y_train = X[idx], y[idx]
    of2.partial_fit(X_train, y_train)



# print('OF:', t2 - t1, 'Acc:', of.score(X_test, y_test))


# t = of.get_nodes(0)
print(of1.n_nodes())
print(of1.n_nodes_reserved())

print(of2.n_nodes())
print(of2.n_nodes_reserved())

# print(t)
