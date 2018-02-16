import os
import pandas as pd
import numpy as np
from time import time
import zipfile

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tick.online import OnlineForestClassifier

import os
import numpy as np
import pandas as pd
import seaborn.apionly as sns
import pylab as pl
import warnings

import matplotlib.pyplot as plt

import sys

sys.path.append('/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/')


from datasets import readers as all_readers

from skgarden import MondrianForestClassifier
from tick.online import OnlineForestClassifier


path = '/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/datasets/'

# path = './datasets/'
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
of.partial_fit(X, y)
t2 = time()
print('OF:', t2 - t1, 'Acc:', of.score(X_test, y_test))


