# License: BSD 3 clause


import numpy as np


class OnlineDummyClassifier(object):
    """A dummy online classifier only using past frequencies of the labels

    Parameters
    ----------
    n_classes : `int`
        Number of classes, we need this information since in a online setting,
        we don't know the number of classes in advance.

    clip : `bool`, default=True
        Scores are clipped whenever too close to zero. Useful whenever log-loss
        is used for evaluation

    eps : `float`, default=1e-15
        Scores smaller than eps are replaces by eps whenever clip==True

    Attributes
    ----------
    n_features : `int`
        The number of features from the training dataset (passed to ``fit``)

    n_samples : `int`
        Number of samples seen
    """

    def __init__(self, n_classes: int, clip: bool = True, eps: float = 1e-15):
        self.n_classes = n_classes
        self.clip = clip
        self.eps = eps
        self.counts = np.zeros((n_classes,), dtype=np.int64)
        self.n_samples = 0

    def partial_fit(self, _, y, classes=None):
        for yi in y:
            self.counts[int(yi)] += 1
            self.n_samples += 1
        return self

    def predict_proba(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix to predict for.

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_classes)
            Returns predicted values.
        """
        n_samples = X.shape[0]
        scores = self.counts / self.n_samples
        probas = np.tile(scores, reps=(n_samples, 1))
        return probas

    def predict(self, X):
        scores = self.predict_proba(X)
        return scores.argmax(axis=1)
