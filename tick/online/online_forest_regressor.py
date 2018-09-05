# License: BSD 3 clause

from abc import ABC

from tick.base import Base
from tick.base import actual_kwargs

from .build.online import OnlineForestRegressor as _OnlineForestRegressor
from tick.preprocessing.utils import safe_array

from .build.online import CriterionRegressor_unif as unif
from .build.online import CriterionRegressor_mse as mse


class OnlineForestRegressor(ABC, Base):
    """Truly online random forest for regression (continuous labels). BLABLA

    Parameters
    ----------
    n_trees : `int`, default=10
        Number of trees to grow in the forest. Cannot be changed after the first
        call to ``fit``.

    criterion : {'unif', 'mse'}, default='unif'
        The criterion used to selected a split. Supported criteria are:
        * 'unif': splits are sampled uniformly in the range of the features, and
          the feature to be splitted is chosen uniformly at random
        * 'mse': the split and feature leading to the best variance reduction
           is selected
        This cannot be changed after the first call to ``fit``

    max_depth : `int`, default=-1
        The maximum depth of a tree. If <= 0, nodes are splitted with no limit
        on the depth of the tree

    min_samples_split : `int`, default=50
        A node waits to contain `min_samples_split` before splitting.

    n_threads : `int`, default=1
        The number of threads used to grow trees in parallel during training.
        If n_threads < 0, then all available cores will be used.

    seed : `int`, default=-1
        If seed >= 0, this is used to seed the random number generators of the
        forest.

    verbose : `bool`, default=True
        If True, then verboses things during training

    warm_start : `bool`, default=True
        If True, then successive calls to ``fit`` will continue to grow existing
        trees. Otherwise, we start from empty trees

    n_splits : `int`, default=10
        Number of potential splits to consider for a feature. BLABLA ???

    Attributes
    ----------
    n_samples : `int`
        Number of samples seen during training

    n_features : int
        The number of features from the training dataset (passed to ``fit``)
    """

    _attrinfos = {
        '_actual_kwargs': {'writable': False},
        '_fitted': {'writable': False},
        '_forest': {'writable': False},
        '_criterion': {'writable': False, 'cpp_setter': 'set_criterion'},
        'n_trees': {'writable': True, 'cpp_setter': 'set_n_trees'},
        'max_depth': {'writable': True, 'cpp_setter': 'set_max_depth'},
        'min_samples_split': {'writable': True,
                              'cpp_setter': 'set_min_samples_split'},
        'n_threads': {'writable': True, 'cpp_setter': 'set_n_threads'},
        'seed': {'writable': True, 'cpp_setter': 'set_seed'},
        'verbose': {'writable': True, 'cpp_setter': 'set_verbose'},
        'warm_start': {'writable': True, 'cpp_setter': 'set_warm_start'},
        'n_splits': {'writable': True, 'cpp_setter': 'set_n_splits'},
    }

    _cpp_obj_name = "_forest"

    @actual_kwargs
    def __init__(self, n_trees: int = 10, step: float = 1.,
                 criterion: str = 'unif',
                 max_depth: int = -1, min_samples_split: int = 50,
                 n_threads: int = 1, seed: int = -1, verbose: bool = True,
                 warm_start: bool = True, n_splits: int = 10):
        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}
        self._fitted = False
        self.n_trees = n_trees
        self.step = step
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_splits = n_splits
        self._forest = _OnlineForestRegressor(n_trees,
                                              step,
                                              self._criterion,
                                              #max_depth,
                                              # min_samples_split,
                                              n_threads,
                                              seed,
                                              verbose)
                                              #warm_start, n_splits)

    def set_data(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        self._forest.set_data(X, y)

    def fit(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        self._set("_fitted", True)
        self._forest.fit(X, y)
        return self

    def apply(self, X):
        """Make the samples from X follow the trees from the forest, and return
        the indices of the leaves
        """
        raise NotImplementedError()

    def predict(self, X, use_aggregation: bool=True):
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
        self._forest.predict(X, y_pred, True)
        return y_pred

    def score(self, X, y):
        from sklearn.metrics import r2_score

    def print(self):
        self._forest._print()

        # TODO: property for splits

    @property
    def criterion(self):
        if self._criterion == unif:
            return 'unif'
        else:
            return 'mse'

    @criterion.setter
    def criterion(self, value):
        if value == 'unif':
            self._set('_criterion', unif)
            # self._forest.set_criterion(unif)
        elif value == 'mse':
            self._set('_criterion', mse)
            # self._forest.set_criterion(mse)
        else:
            raise ValueError("``criterion`` must be either 'unif' or 'mse'.")
