
import numpy as np
from sklearn.model_selection import train_test_split
# from skgarden import MondrianForestClassifier, MondrianTreeClassifier

from tick.online import OnlineForestClassifier

import sys
sys.path.append('/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/')

path = '/Users/stephane.gaiffas/Dropbox/jaouad/code/online-forests/datasets/'

from datasets import readers as all_readers

readers = all_readers

X, y, dataset_name = readers[13](path)
n_samples, n_features = X.shape
n_classes = int(y.max() + 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=.3, random_state=42)


# mf = MondrianForestClassifier(n_estimators=10)
# mf.partial_fit(X_train, y_train, classes=np.arange(n_classes))
# mf.apply(X_test).max(axis=1).max()
# mf_paths, mf_est_inds = mf.weighted_decision_path(X_test)
# mf_paths.shape, mf_est_inds.shape
# i = 0
# mf_paths[:, mf_est_inds[i]: mf_est_inds[i + 1]]

of1 = OnlineForestClassifier(n_classes=n_classes, seed=123,
                             use_aggregation=True,
                             n_trees=10,
                             dirichlet=0.5, step=1.,
                             use_feature_importances=False)

# of2 = OnlineForestClassifier(n_classes=n_classes, seed=123,
#                              use_aggregation=False,
#                              n_trees=10,
#                              dirichlet=0.5, step=1.,
#                              use_feature_importances=False)

of1.partial_fit(X_train, y_train)

# of2.partial_fit(X_train, y_train)

np.set_printoptions(precision=2)

# i = 123
#
# plt.subplot(1, 3, 1)
# plt.stem(mf.predict_proba(X_test[i, :].reshape(1, n_features)).ravel())
# plt.ylim((0, 1))
# plt.subplot(1, 3, 2)
# plt.stem(of1.predict_proba(X_test[i, :].reshape(1, n_features)).ravel())
# plt.title('with aggregation')
# plt.ylim((0, 1))
# plt.subplot(1, 3, 3)
# plt.stem(of2.predict_proba(X_test[i, :].reshape(1, n_features)).ravel())
# plt.title('No aggregation')
# plt.ylim((0, 1))
#
# plt.tight_layout()



for i in range(10):
    # print(of1.get_path_depth(0, X_test[i, :].ravel()))
    print(of1.get_path(0, X_test[i, :].ravel()))

