# License: BSD 3 clause

from abc import ABC

from tick.base import Base
from tick.base import actual_kwargs

from .build.inference import OnlineForest as _OnlineForest
from tick.preprocessing.utils import safe_array

from .build.inference import Criterion_unif as as unif
from .build.model import Criterion_mse as mse


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

    max_depth : `int` or `None`, default=None
        The maximum depth of a tree. If None, nodes are splitted until expanded
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : `int`, default=50
        A node waits to contain `min_samples_split` before splitting.

    n_threads : `int`, default=1
        The number of threads used to grow trees in parallel during training.
        If n_threads < 0, then all available cores will be used.

    seed: `int` or `None`, default=None
        If int, this is used to seed the random number generators of the forest.

    verbose : `bool`, default=True
        If True, then verboses things during training

    warm_start : `bool`, default=True
        If True, then successive calls to ``fit`` will continue to grow existing
        trees. Otherwise, we start from empty trees

    n_splits : `int` , default=10
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
        '_criterion': {'writable': False},
        'n_trees': {'writable': True, 'cpp_setter': 'set_n_trees'},
        'criterion': {'writable': True, 'cpp_setter': 'set_criterion'},
        'max_depth': {'writable': True, 'cpp_setter': 'set_max_depth'},
        'min_samples_split': {'writable': True, 'cpp_setter': 'set_min_samples_split'},
        'n_threads': {'writable': True, 'cpp_setter': 'set_n_threads'},
        'seed': {'writable': True, 'cpp_setter': 'set_seed'},
        'verbose': {'writable': True, 'cpp_setter': 'set_verbose'},
        'warm_start': {'writable': True, 'cpp_setter': 'set_warm_start'},
        'n_splits': {'writable': True, 'cpp_setter': 'set_n_splits'},
    }

    _cpp_obj_name = "_forest"

    @actual_kwargs
    def __init__(self, n_trees: int = 10, criterion: str = 'unif',
                 max_depth: int = None, min_samples_split: int = 50,
                 n_threads: int = 1, seed: int = None, verbose: bool = True,
                 warm_start: bool = True, n_splits: int = 10):
        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}
        self._fitted = False
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_threads = n_threads
        self.seed = seed
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_splits = n_splits
        self._forest = _OnlineForest(n_trees, self._criterion, max_depth,
                                     min_samples_split, n_threads, seed,
                                     verbose, warm_start, n_splits)

    def set_data(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        self._forest.set_data(X, y)

    def fit(self, n_iter=0):
        self._set("_fitted", True)
        self._forest.fit(n_iter)
        return self

    def apply(self, X):
        """Make the samples from X follow the trees from the forest, and return
        the indices of the leaves

        """
        raise NotImplementedError()

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

    # TODO: property for splits

    # @property
    # def link(self):
    #     return self._link
    #
    # @link.setter
    # def link(self, value):
    #     if self._link is not None:
    #         raise ValueError("link is read only")
    #     if value == "exponential":
    #         self._set("_link_type", exponential)
    #     elif value == "identity":
    #         self._set("_link_type", identity)
    #     else:
    #         raise ValueError("``link`` must be either 'exponential' or "
    #                          "'linear'.")
    #     self._set("_link", value)
