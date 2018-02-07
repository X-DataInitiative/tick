
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
from tick.online import OnlineForestClassifier
from tick.simulation import weights_sparse_exp
from tick.linear_model import SimuLogReg
from sklearn.ensemble import RandomForestClassifier


np.set_printoptions(precision=2)

n_samples = 10000
n_features = 30
n_classes = 2

nnz = 5
w0 = np.zeros(n_features)
w0[:nnz] = 1

# TODO: Seed

n_trees = 50

# w0 = weights_sparse_exp(n_features, nnz=nnz)

X, y = SimuLogReg(weights=w0, intercept=None, n_samples=n_samples,
                  cov_corr=0.1, features_scaling='standard',
                  seed=123).simulate()
y = (y + 1) / 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

rf = RandomForestClassifier(n_estimators=n_trees, criterion="entropy",
                            random_state=123)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_

of1 = OnlineForestClassifier(n_classes=n_classes, n_trees=n_trees, seed=123,
                             step=1., use_aggregation=True,
                             use_feature_importances=True)
of1.fit(X, y)

of2 = OnlineForestClassifier(n_classes=n_classes, n_trees=n_trees, seed=123,
                             step=1., use_aggregation=True,
                             use_feature_importances=feature_importances)
of2.fit(X, y)


# print('*' * 32, 'true')
# of = OnlineForestClassifier(n_classes=n_classes, n_trees=10, seed=123,
#                              step=1., use_aggregation=True,
#                             use_feature_importances=True)
# of.fit(X, y)
# print("of.use_feature_importances", of.use_feature_importances)
# print("of._feature_importances_type", of._feature_importances_type)
# print("of._given_feature_importances", of._given_feature_importances)
# print("of.feature_importances", of.feature_importances)
#
# print('*' * 32, 'false')
# of = OnlineForestClassifier(n_classes=n_classes, n_trees=10, seed=123,
#                              step=1., use_aggregation=True,
#                             use_feature_importances=False)
# of.fit(X, y)
# print("of.use_feature_importances", of.use_feature_importances)
# print("of._feature_importances_type", of._feature_importances_type)
# print("of._given_feature_importances", of._given_feature_importances)
# print("of.feature_importances", of.feature_importances)
#
# print('*' * 32, 'given')
# of = OnlineForestClassifier(n_classes=n_classes, n_trees=10, seed=123,
#                              step=1., use_aggregation=True,
#                             use_feature_importances=feature_importances)
# of.fit(X, y)
# print("of.use_feature_importances", of.use_feature_importances)
# print("of._feature_importances_type", of._feature_importances_type)
# print("of._given_feature_importances", of._given_feature_importances)
# print("of.feature_importances", of.feature_importances)


#
#
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#
# of.fit(X_train, y_train)
#
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred_of1 = of1.predict_proba(X_test)[:, 1]
y_pred_of2 = of2.predict_proba(X_test)[:, 1]

print("AUC rf=", roc_auc_score(y_test, y_pred_rf))
print("AUC of1=", roc_auc_score(y_test, y_pred_of1))
print("AUC of2=", roc_auc_score(y_test, y_pred_of2))


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.stem(rf.feature_importances_)
plt.title('Breiman Random Forest', fontsize=16)
plt.subplot(1, 4, 2)
plt.stem(of1.feature_importances)
plt.title('Online Forest', fontsize=16)
plt.subplot(1, 4, 3)
plt.stem(of2.feature_importances)
plt.title('Online Forest', fontsize=16)

plt.subplot(1, 4, 4)
plt.stem(w0)
plt.title('Model weights', fontsize=16)

plt.show()
plt.tight_layout()

exit(0)
# print(clf.feature_importances)
