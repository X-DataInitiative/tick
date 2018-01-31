"""
==============================================
Binary classification with logistic regression
==============================================

This code perform binary classification on adult dataset with logistic
regression learner (`tick.inference.LogisticRegression`).
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tick.linear_model import LogisticRegression
from tick.dataset import fetch_tick_dataset

train_set = fetch_tick_dataset('binary/adult/adult.trn.bz2')
test_set = fetch_tick_dataset('binary/adult/adult.tst.bz2')

learner = LogisticRegression()
learner.fit(train_set[0], train_set[1])

predictions = learner.predict_proba(test_set[0])
fpr, tpr, _ = roc_curve(test_set[1], predictions[:, 1])

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2)
plt.title("ROC curve on adult dataset (area = {:.2f})".format(auc(fpr, tpr)))
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

plt.show()
