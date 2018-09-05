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


# TODO: respect the scikit-learn API: partial_fit with classes= ?
# TODO: clean the C++ part, optimize a little bit the code (memory allocation)
# TODO: add a random_state argument to the classifier

# TODO: max_nodes_memorized_range = memory / (8 * n_trees * n_features) where memory is in Bytes


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
    n_features : int
        The number of features from the training dataset (passed to ``fit``)
    """

    _attrinfos = {
        '_actual_kwargs': {'writable': False},
        '_fitted': {'writable': False},
        '_forest': {'writable': False},
        '_memory': {'writable': False},
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

    @actual_kwargs
    def __init__(self, n_classes: int, n_trees: int = 10, step: float = 1.,
                 criterion: str = 'log', use_aggregation: bool = True,
                 dirichlet: float = None, split_pure: bool = False,
                 max_nodes: int = None, min_extension_size: float = 0,
                 min_samples_split: int=-1, max_features: int=-1,
                 n_threads: int = 1,
                 use_feature_importances=True, seed: int = -1,
                 verbose: bool = True, print_every=1000, memory: int = 512):
        Base.__init__(self)
        if not hasattr(self, "_actual_kwargs"):
            self._actual_kwargs = {}
        self._fitted = False
        self.n_trees = n_trees
        self.n_features = None
        self.n_classes = n_classes
        self.step = step
        self.criterion = criterion
        self.split_pure = split_pure
        if max_nodes is not None:
            self.max_nodes = max_nodes
        else:
            self.max_nodes = -1
        self.min_extension_size = min_extension_size
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_threads = n_threads
        self._forest = None
        self._given_feature_importances = None
        self._feature_importances_type = None
        self.use_feature_importances = use_feature_importances
        self.seed = seed
        self.verbose = verbose
        self.print_every = print_every
        self.use_aggregation = use_aggregation
        self._forest = None

        if dirichlet is None:
            if self.n_classes == 2:
                self.dirichlet = 0.5
            else:
                self.dirichlet = 0.01
        else:
            self.dirichlet = dirichlet

        self._set('_memory', memory)

    def set_data(self, X, y):
        X = safe_array(X, dtype='float32')
        y = safe_array(y, dtype='float32')
        self._forest.set_data(X, y)

    def partial_fit(self, X, y, classes=None):
        """

        :param X:
        :param y:
        :param classes:
        :return:
        """
        X = safe_array(X, dtype='float32')
        y = safe_array(y, dtype='float32')
        n_samples, n_features = X.shape
        # TODO: check that sizes of X and y match
        if self._forest is None:
            self.n_features = n_features
            # print(f"n_features: {n_features}, n_trees: {self.n_trees}")
            max_nodes_with_memory_in_tree \
                = int(1024 ** 2 * self.memory / (8 * self.n_trees * n_features))

            # max_nodes_with_memory_in_tree = 20000

            _forest = _OnlineForestClassifier(
                n_features,
                self.n_classes,
                self.n_trees,
                self.step,
                self._criterion,
                self._feature_importances_type,
                self.use_aggregation,
                self.dirichlet,
                self.split_pure,
                self.max_nodes,
                self.min_extension_size,
                self.min_samples_split,
                self.max_features,
                self.n_threads,
                self.seed,
                self.verbose,
                self.print_every,
                max_nodes_with_memory_in_tree
                # self.verbose_every
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
        scores = np.empty((X.shape[0], self.n_classes), dtype='float32')
        if not self._fitted:
            raise RuntimeError("You must call ``fit`` before")
        else:
            X = safe_array(X, dtype='float32')
        self._forest.predict(X, scores)
        return scores

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("You must call ``fit`` before")
        else:
            X = safe_array(X, dtype='float32')
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

    def n_nodes_reserved(self):
        n_nodes_reserved_per_tree = np.empty(self.n_trees, dtype=np.uint32)
        self._forest.n_nodes_reserved(n_nodes_reserved_per_tree)
        return n_nodes_reserved_per_tree

    @property
    def memory(self):
        return self._memory

    # TODO: no setter for memory

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

    def get_path(self, n_tree, xt):
        self._forest.get_path(n_tree, xt)

    def get_path_depth(self, n_tree, x_t):
        return self._forest.get_path_depth(n_tree, x_t)

    def get_path(self, n_tree, x_t):
        depth = self.get_path_depth(n_tree, x_t)
        path = np.empty(depth, dtype=np.uint32)
        self._forest.get_path(n_tree, x_t, path)
        return path

    def get_nodes_json(self, tree):
        n_nodes = self.n_nodes()[tree]
        # print("n_nodes=", n_nodes)
        nodes_parent = np.empty(n_nodes, dtype=np.uint32)
        nodes_left = np.empty(n_nodes, dtype=np.uint32)
        nodes_right = np.empty(n_nodes, dtype=np.uint32)
        nodes_feature = np.empty(n_nodes, dtype=np.uint32)
        nodes_threshold = np.empty(n_nodes, dtype=np.float32)
        nodes_time = np.empty(n_nodes, dtype=np.float32)
        nodes_features_min = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_features_max = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_n_samples = np.empty(n_nodes, dtype=np.uint32)
        nodes_weight = np.empty(n_nodes, dtype=np.float32)
        nodes_weight_tree = np.empty(n_nodes, dtype=np.float32)
        nodes_is_leaf = np.empty(n_nodes, dtype=np.ushort)
        nodes_counts = np.empty((n_nodes, self.n_classes), dtype=np.uint32)

        self._forest.get_flat_nodes(
            tree,
            nodes_parent,
            nodes_left,
            nodes_right,
            nodes_feature,
            nodes_threshold,
            nodes_time,
            nodes_features_min,
            nodes_features_max,
            nodes_n_samples,
            nodes_weight,
            nodes_weight_tree,
            nodes_is_leaf,
            nodes_counts)
        nodes = []
        for index, (parent, left, right, time) \
                in enumerate(zip(nodes_parent, nodes_left,
                                 nodes_right, nodes_time)):
            nodes.append(
                {'index': int(index), 'left': int(left), 'right': int(right),
                 'time': float(time)}
            )
        return nodes

    def get_nodes_df(self, tree):
        import pandas as pd
        n_nodes = self.n_nodes()[tree]
        # print("n_nodes=", n_nodes)
        nodes_parent = np.empty(n_nodes, dtype=np.uint32)
        nodes_left = np.empty(n_nodes, dtype=np.uint32)
        nodes_right = np.empty(n_nodes, dtype=np.uint32)
        nodes_feature = np.empty(n_nodes, dtype=np.uint32)
        nodes_threshold = np.empty(n_nodes, dtype=np.float32)
        nodes_time = np.empty(n_nodes, dtype=np.float32)
        nodes_depth = np.empty(n_nodes, dtype=np.ushort)
        nodes_features_min = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_features_max = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_n_samples = np.empty(n_nodes, dtype=np.uint32)
        nodes_sample = np.empty(n_nodes, dtype=np.uint32)
        nodes_weight = np.empty(n_nodes, dtype=np.float32)
        nodes_weight_tree = np.empty(n_nodes, dtype=np.float32)
        nodes_is_leaf = np.empty(n_nodes, dtype=np.ushort)
        nodes_is_memorized = np.empty(n_nodes, dtype=np.ushort)
        nodes_counts = np.empty((n_nodes, self.n_classes), dtype=np.uint32)

        self._forest.get_flat_nodes(
            tree,
            nodes_parent,
            nodes_left,
            nodes_right,
            nodes_feature,
            nodes_threshold,
            nodes_time,
            nodes_depth,
            nodes_features_min,
            nodes_features_max,
            nodes_n_samples,
            nodes_sample,
            nodes_weight,
            nodes_weight_tree,
            nodes_is_leaf,
            nodes_is_memorized,
            nodes_counts)

        index = np.arange(n_nodes)
        columns = ['id', 'parent', 'left', 'right', 'depth', 'leaf',
                   'feature', 'threshold', 'time', 'n_samples', 'sample',
                   'features_min', 'features_max', 'memorized']
        data = {'id': index, 'parent': nodes_parent, 'left': nodes_left,
                'right': nodes_right, 'depth': nodes_depth,
                'feature': nodes_feature,
                'threshold': nodes_threshold,
                'leaf': nodes_is_leaf.astype(np.bool),
                'memorized': nodes_is_memorized.astype(np.bool),
                'time': nodes_time, 'n_samples': nodes_n_samples,
                'sample': nodes_sample,
                'features_min': [tuple(t) for t in nodes_features_min],
                'features_max': [tuple(t) for t in nodes_features_max]}
        df = pd.DataFrame(data, columns=columns)
        return df

    def get_nodes(self, tree):
        n_nodes = self.n_nodes()[tree]
        # print("n_nodes=", n_nodes)
        nodes_parent = np.empty(n_nodes, dtype=np.uint32)
        nodes_left = np.empty(n_nodes, dtype=np.uint32)
        nodes_right = np.empty(n_nodes, dtype=np.uint32)
        nodes_feature = np.empty(n_nodes, dtype=np.uint32)
        nodes_threshold = np.empty(n_nodes, dtype=np.float32)
        nodes_time = np.empty(n_nodes, dtype=np.float32)
        nodes_depth = np.empty(n_nodes, dtype=np.ushort)
        nodes_features_min = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_features_max = np.empty((n_nodes, self.n_features),
                                      dtype=np.float32)
        nodes_n_samples = np.empty(n_nodes, dtype=np.uint32)
        nodes_sample = np.empty(n_nodes, dtype=np.uint32)
        nodes_weight = np.empty(n_nodes, dtype=np.float32)
        nodes_weight_tree = np.empty(n_nodes, dtype=np.float32)
        nodes_is_leaf = np.empty(n_nodes, dtype=np.ushort)
        nodes_counts = np.empty((n_nodes, self.n_classes), dtype=np.uint32)

        self._forest.get_flat_nodes(
            tree,
            nodes_parent,
            nodes_left,
            nodes_right,
            nodes_feature,
            nodes_threshold,
            nodes_time,
            nodes_depth,
            nodes_features_min,
            nodes_features_max,
            nodes_n_samples,
            nodes_sample,
            nodes_weight,
            nodes_weight_tree,
            nodes_is_leaf,
            nodes_counts)

        index = np.arange(n_nodes)
        nodes_info = {'index': index, 'parent': nodes_parent,
                      'left': nodes_left,
                      'right': nodes_right, 'depth': nodes_depth,
                      'feature': nodes_feature,
                      'threshold': nodes_threshold,
                      'leaf': nodes_is_leaf.astype(np.bool),
                      'time': nodes_time, 'n_samples': nodes_n_samples,
                      'sample': nodes_sample}
        return nodes_info

    def print_tree(self, n_tree):
        nodes_info = self.get_nodes(n_tree)
        indexes = nodes_info['index']
        parents = nodes_info['parent']
        lefts = nodes_info['left']
        rights = nodes_info['right']
        depths = nodes_info['depth']
        leafs = nodes_info['leaf']
        times = nodes_info['time']
        n_samples = nodes_info['n_samples']

        depths[0] = 0
        max_depth = depths.max()
        for depth in range(max_depth):
            print('=' * 16)
            print('depth:', depth)
            print('-' * 8)
            filt = (depths == depth)
            print('index:', indexes[filt])
            print('leaf:   ', leafs[filt].astype(np.int))
            print('parent:', parents[filt])
            print('left:   ', lefts[filt])
            print('right:   ', rights[filt])
            print('time:   ', times[filt])
            print('n_samples:   ', n_samples[filt])

    def n_samples(self):
        return self._forest.n_samples()
