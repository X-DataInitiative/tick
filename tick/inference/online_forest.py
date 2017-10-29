# License: BSD 3 clause

from abc import ABC

from tick.base import Base
from tick.base import actual_kwargs

from .build.inference import OnlineForest as _OnlineForest
from tick.preprocessing.utils import safe_array

class OnlineForest(ABC, Base):
    """Truly online random forest for regression (continuous labels).

    Parameters
    ----------
    max_depth : `int`
        Maximum depth of the trees

    Other Parameters
    ----------------

    Attributes
    ----------
    """

    _attrinfos = {
        "_actual_kwargs": {"writable": False},
        "_fitted": {"writable": False},
        "_forest": {"writable": False}
    }

    @actual_kwargs
    def __init__(self, n_trees: int = 100, n_min_samples: int = 20,
                 n_splits: int=10):
        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}
        self._fitted = False
        self.n_trees = n_trees
        self.n_min_samples = n_min_samples
        self.n_splits = n_splits
        self._forest = _OnlineForest(n_trees, n_min_samples, n_splits)

    def set_data(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        self._forest.set_data(X, y)

    def fit(self, n_iter=0):
        self._set("_fitted", True)
        self._forest.fit(n_iter)
        return self

    def predict(self, X):
        """Predict class for given samples

        Parameters
        ----------
        X : `np.ndarray` or `scipy.sparse.csr_matrix`, shape=(n_samples, n_features)
            Features matrix to predict for.

        Returns
        -------
        output : `np.array`, shape=(n_samples,)
            Returns predicted values.
        """
        import numpy as np
        y_pred = np.empty(X.shape[0])
        if not self._fitted:
            raise ValueError("You must call ``fit`` before")
        else:
            X = safe_array(X)
        self._forest.predict(X, y_pred)
        return y_pred

    def score(self, X, y):
        from sklearn.metrics import r2_score

    def print(self):
        self._forest._print()
