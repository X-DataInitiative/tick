# License: BSD 3 clause

from abc import ABC

import numpy as np

from tick.base import Base
from tick.base import actual_kwargs
from tick.preprocessing.utils import safe_array

from .build.online import OnlineForestClassifier as _OnlineForestClassifier
from .build.online import CriterionClassifier_log

from .build.online import FeatureImportanceType_no
from .build.online import FeatureImportanceType_estimated
from .build.online import FeatureImportanceType_given


class OnlineForestClassifier(ABC, Base):
    """Truly online random forest for regression (continuous labels). BLABLA

    TODO: update docstrings

    Parameters
    ----------
    n_classes : `int`
        Number of classes, we need this information since in a online setting,
        we don't know the number of classes in advance.

    n_trees : `int`, default=10
        Number of trees to grow in the forest. Cannot be changed after the first
        call to ``fit``.

    step : `float`, default=1.
        Step-size for the aggregation weights. Default is 1 for classification.

    criterion : {'log'}, default='log'
        The criterion used to selected a split. Supported criteria are:
        * 'unif': splits are sampled uniformly in the range of the features, and
          the feature to be splitted is chosen uniformly at random
        * 'mse': the split and feature leading to the best variance reduction
           is selected
        This cannot be changed after the first call to ``fit``

    use_aggregation : `bool`, default=True
        If True

    use_feature_importances : `bool` or `np.array`
        If `True`, then we estimate the feature importances online, if `False`
        all the features are used. If a `numpy.array` is given, then it is a
        vector of probabilities that give the importance of each feature.

    n_threads : `int`, default=1
        The number of threads used to grow trees in parallel during training.
        If n_threads < 0, then all available cores will be used.

    seed : `int`, default=-1
        If seed >= 0, this is used to seed the random number generators of the
        forest.

    verbose : `bool`, default=True
        If True, then verboses things during training

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
        'n_threads': {'writable': True, 'cpp_setter': 'set_n_threads'},
        'seed': {'writable': True, 'cpp_setter': 'set_seed'},
        '_use_feature_importances': {'writable': False},
        '_feature_importances_type': {'writable': False},
        '_given_feature_importances': {'writable': False},
        'verbose': {'writable': True, 'cpp_setter': 'set_verbose'},
        'warm_start': {'writable': True, 'cpp_setter': 'set_warm_start'},
        # 'n_splits': {'writable': True, 'cpp_setter': 'set_n_splits'},
        'dirichlet': {'writable': True, 'cpp_setter': 'set_dirichlet'}
    }

    _cpp_obj_name = "_forest"

    # TODO: n_classes must be mandatory

    @actual_kwargs
    def __init__(self, n_classes: int, n_trees: int = 10, n_passes: int = 1,
                 step: float = 1.,
                 criterion: str = 'log', use_aggregation: bool = True,
                 subsampling: float=1., dirichlet: float=None,
                 n_threads: int = 1, use_feature_importances=True,
                 seed: int = -1, verbose: bool = True):
        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}
        self._fitted = False
        self.n_trees = n_trees
        self.n_passes = n_passes
        self.n_features = None
        self.n_classes = n_classes
        self.step = step
        self.criterion = criterion
        self.n_threads = n_threads
        self._forest = None
        self._given_feature_importances = None
        self._feature_importances_type = None
        self.use_feature_importances = use_feature_importances
        self.seed = seed
        self.verbose = verbose
        self.use_aggregation = use_aggregation
        self.subsampling = subsampling
        self._forest = None
        if dirichlet is None:
            dirichlet = 1 / n_classes
        self.dirichlet = dirichlet

    def set_data(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        self._forest.set_data(X, y)

    def fit(self, X, y):
        X = safe_array(X)
        y = safe_array(y)
        n_samples, n_features = X.shape
        # TODO: check that sizes of X and y match
        if self._forest is None:
            self.n_features = n_features
            _forest = _OnlineForestClassifier(
                n_features, self.n_classes, self.n_trees, self.n_passes,
                self.step,
                self._criterion, self._feature_importances_type,
                self.use_aggregation, self.subsampling,
                self.dirichlet, self.n_threads, self.seed, self.verbose
            )
            if self._feature_importances_type == FeatureImportanceType_given:
                _forest.set_given_feature_importances(
                    self._given_feature_importances)
            self._set('_forest', _forest)
        self._set("_fitted", True)
        self._forest.fit(X, y)
        return self

    def apply(self, X):
        """Make the samples from X follow the trees from the forest, and return
        the indices of the leaves
        """
        raise NotImplementedError()

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
        import numpy as np
        scores = np.empty((X.shape[0], self.n_classes))
        if not self._fitted:
            raise RuntimeError("You must call ``fit`` before")
        else:
            X = safe_array(X)
        self._forest.predict(X, scores)
        return scores

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("You must call ``fit`` before")
        else:
            scores = self.predict_proba(X)
            return scores.argmax(axis=1)

    def clear(self):
        if self._forest is not None:
            self._forest.clear()

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def print(self):
        self._forest._print()
        # TODO: property for splits

    def n_leaves(self):
        n_leaves_per_tree = np.empty(self.n_trees, dtype=np.uint32)
        self._forest.n_leaves(n_leaves_per_tree)
        return n_leaves_per_tree

    def n_nodes(self):
        n_nodes_per_tree = np.empty(self.n_trees, dtype=np.uint32)
        self._forest.n_nodes(n_nodes_per_tree)
        return n_nodes_per_tree

    @property
    def criterion(self):
        if self._criterion == CriterionClassifier_log:
            return 'log'

    @criterion.setter
    def criterion(self, value):
        if value == 'log':
            self._set('_criterion', CriterionClassifier_log)
        else:
            raise ValueError("``criterion`` must be either 'unif' or 'mse'.")

    @property
    def use_feature_importances(self):
        feature_importances_type = self._feature_importances_type
        if feature_importances_type == FeatureImportanceType_no:
            return False
        elif feature_importances_type == FeatureImportanceType_estimated:
            return True
        else:
            return self._given_feature_importances

    @use_feature_importances.setter
    def use_feature_importances(self, value):
        if type(value) is bool:
            if value:
                self._set('_feature_importances_type',
                          FeatureImportanceType_estimated)
            else:
                self._set('_feature_importances_type',
                          FeatureImportanceType_no)
        elif type(value) is np.ndarray:
            self._set('_feature_importances_type',
                      FeatureImportanceType_given)
            self._set('_given_feature_importances', value)
            if self._forest is not None:
                self._forest.set_feature_importances(value)
        else:
            raise ValueError('use_feature_importances can be either `bool` or'
                             'a numpy array')

    @property
    def feature_importances(self):
        if self._feature_importances_type == FeatureImportanceType_no:
            return None
        else:
            if self._forest is None:
                return None
            else:
                feature_importances = np.empty(self.n_features)
                self._forest.get_feature_importances(feature_importances)
                return feature_importances
