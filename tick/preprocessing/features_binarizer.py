# License: BSD 3 clause

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from tick.base import Base


class FeaturesBinarizer(Base, BaseEstimator, TransformerMixin):
    """Transforms continuous data into bucketed binary data.

    This is a scikit-learn transformer that transform an input
    pandas DataFrame X of shape (n_samples, n_features) into a binary
    matrix of size (n_samples, n_new_features).
    Continous features are modified and extended into binary features, using
    linearly or inter-quantiles spaced bins.
    Discrete features are binary encoded with K columns, where K is the number
    of modalities.
    Other features (none of the above) are left unchanged.

    Parameters
    ----------
    n_cuts : `int`, default=10
        Number of cut points for continuous features.

    method : "quantile" or "linspace", default="quantile"
        * If ``"quantile"`` quantile-based cuts are used.
        * If ``"linspace"`` linearly spaced cuts are used.
        * If ``"given"`` bins_boundaries needs to be provided.

    detect_column_type : "auto" or "column_names", default="auto"
        * If ``"auto"`` feature type detection done automatically.
        * If ``"column_names"`` feature type detection done using column names.
          In this case names ending by ":continuous" means continuous
          while ":discrete" means a discrete feature

    remove_first : `bool`
        If `True`, first column of each binarized continuous feature block is
        removed.

    bins_boundaries : `list`, default="none"
        Bins boundaries for continuous features.

    Attributes
    ----------
    one_hot_encoder : `OneHotEncoder`
        OneHotEncoders for continuous and discrete features.

    bins_boundaries : `list`
        Bins boundaries for continuous features.

    mapper : `dict`
        Map modalities to column indexes for categorical features.

    feature_type : `dict`
        Features type.

    blocks_start : `list`
        List of indices of the beginning of each block of binarized features

    blocks_length : `list`
        Length of each block of binarized features

    References
    ----------
    http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

    Examples
    --------
    >>> import numpy as np
    >>> from tick.preprocessing import FeaturesBinarizer
    >>> features = np.array([[0.00902084, 0., 'z'],
    ...                      [0.46599565, 0., 2.],
    ...                      [0.52091721, 1., 2.],
    ...                      [0.47315496, 1., 1.],
    ...                      [0.08180209, 0., 0.],
    ...                      [0.45011727, 0., 0.],
    ...                      [2.04347947, 1., 20.],
    ...                      [-0.9890938, 0., 0.],
    ...                      [-0.3063761, 1., 1.],
    ...                      [0.27110903, 0., 0.]])
    >>> binarizer = FeaturesBinarizer(n_cuts=3)
    >>> binarized_features = binarizer.fit_transform(features)
    >>> # output comes as a sparse matrix
    >>> binarized_features.__class__
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> # column type is automatically detected
    >>> sorted(binarizer.feature_type.items())
    [('0', 'continuous'), ('1', 'discrete'), ('2', 'discrete')]
    >>> # features is binarized (first column is removed to avoid colinearity)
    >>> binarized_features.toarray()
    array([[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
           [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
           [0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.]])
    """

    _attrinfos = {
        "one_hot_encoder": {
            "writable": False
        },
        "bins_boundaries": {
            "writable": False
        },
        "mapper": {
            "writable": False
        },
        "feature_type": {
            "writable": False
        },
        "_fitted": {
            "writable": False
        }
    }

    def __init__(self, method="quantile", n_cuts=10, detect_column_type="auto",
                 remove_first=False, bins_boundaries=None):
        Base.__init__(self)

        self.method = method
        self.n_cuts = n_cuts
        self.detect_column_type = detect_column_type
        self.remove_first = remove_first
        self.bins_boundaries = bins_boundaries
        self.reset()

    def reset(self):
        self._set("one_hot_encoder", OneHotEncoder(sparse=True))
        self._set("mapper", {})
        self._set("feature_type", {})
        self._set("_fitted", False)
        if self.method != "given":
            self._set("bins_boundaries", {})

    @property
    def boundaries(self):
        """Get bins boundaries for all features.

        Returns
        -------
        output : `dict`
            The bins boundaries for each feature.
        """
        if not self._fitted:
            raise ValueError("cannot get bins_boundaries if object has not "
                             "been fitted")
        return self.bins_boundaries

    @property
    def blocks_start(self):
        """Get the first column indices of each binarized feature blocks.

        Returns
        -------
        output : `np.ndarray`
            The indices of the first column of each binarized feature blocks.
        """
        if not self._fitted:
            raise ValueError("cannot get blocks_start if object has not "
                             "been fitted")
        # construct from encoder
        return self._get_feature_indices()[:-1,]

    @property
    def blocks_length(self):
        """Get the length of each binarized feature blocks.

        Returns
        -------
        output : `np.ndarray`
            The length of each binarized feature blocks.
        """
        if not self._fitted:
            raise ValueError("cannot get blocks_length if object has not been "
                             "fitted")
        # construct from encoder
        return self._get_n_values()

    @staticmethod
    def cast_to_array(X):
        """Cast input matrix to `np.ndarray`.

        Returns
        -------
        output : `np.ndarray`, `np.ndarray`
            The input matrix and the corresponding column names.
        """
        if X.__class__ == pd.DataFrame:
            columns = X.columns
            X = X.values
        else:
            columns = [str(i) for i in range(X.shape[1])]

        return X, columns

    def fit(self, X):
        """Fit the binarization using the features matrix.

        Parameters
        ----------
        X : `pd.DataFrame`  or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `FeaturesBinarizer`
            The fitted current instance.
        """
        self.reset()
        X, columns = FeaturesBinarizer.cast_to_array(X)
        categorical_X = np.empty_like(X)
        for i, column in enumerate(columns):
            feature = X[:, i]
            binarized_feat = self._assign_interval(column, feature, fit=True)
            categorical_X[:, i] = binarized_feat

        self.one_hot_encoder.fit(categorical_X)

        self._set("_fitted", True)
        return self

    def transform(self, X):
        """Apply the binarization to the given features matrix.

        Parameters
        ----------
        X : `pd.DataFrame` or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        X, columns = FeaturesBinarizer.cast_to_array(X)

        categorical_X = np.empty_like(X)
        for i, column in enumerate(columns):
            feature = X[:, i]
            binarized_feat = self._assign_interval(columns[i], feature,
                                                   fit=False)
            categorical_X[:, i] = binarized_feat

        binarized_X = self.one_hot_encoder.transform(categorical_X)

        if self.remove_first:
            feature_indices = self._get_feature_indices()
            mask = np.ones(binarized_X.shape[1], dtype=bool)
            mask[feature_indices[:-1]] = False
            binarized_X = binarized_X[:, mask]

        return binarized_X

    def fit_transform(self, X, y=None, **kwargs):
        """Fit and apply the binarization using the features matrix.

        Parameters
        ----------
        X : `pd.DataFrame` or `np.ndarray`, shape=(n_samples, n_features)
            The features matrix.

        Returns
        -------
        output : `pd.DataFrame`
            The binarized features matrix. The number of columns is
            larger than n_features, smaller than n_cuts * n_features,
            depending on the actual number of columns that have been
            binarized.
        """
        self.fit(X)
        binarized_X = self.transform(X)

        return binarized_X

    @staticmethod
    def _detect_feature_type(feature, detect_column_type="auto",
                             feature_name=None, continuous_threshold="auto"):
        """Detect the type of a single feature.

        Parameters
        ----------
        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        detect_column_type : "auto" or "column_names", default="auto"
            * If ``"auto"`` an automatic type detection procedure is followed.
            * If ``"column_names"`` columns with name ending with
            ":continuous" means continuous features and columns with name ending
            with ":discrete" means discrete features

        feature_name : `str`
            The name of the feature

        continuous_threshold : `int` or `str`, default "auto"
            If "auto", we consider the feature as "discrete" if the feature
            gets more than `threshold`=15 distinct values (if there are more
            than 30 examples, else `threshold` is set to half the number of
            examples).
            If a number is given, then we consider the feature as "discrete" if
            the feature has more distinct values than this number

        Returns
        -------
        output : `str`
            The type of the feature (either `continuous` or `discrete`).
        """
        if detect_column_type == "column_names":
            if feature_name is None:
                raise ValueError("feature_name must be set in order to use"
                                 "'column_names' detection type")

            if feature_name.endswith(":continuous"):
                feature_type = "continuous"
            elif feature_name.endswith(":discrete"):
                feature_type = "discrete"
            else:
                raise ValueError("feature name '%s' should end with "
                                 "':continuous' or ':discrete'" % feature_name)

        elif detect_column_type == "auto":
            if continuous_threshold == "auto":
                # threshold choice depending on whether one has more than 30
                # examples or not
                if len(feature) > 30:
                    threshold = 15
                else:
                    threshold = len(feature) / 2
            else:
                threshold = continuous_threshold

            # count distinct realizations and compare to threshold
            uniques = np.unique(feature)
            n_uniques = len(uniques)
            if n_uniques > threshold:
                # feature_type is `continuous` only is all feature values are
                # convertible to float
                try:
                    uniques.astype(float)
                    feature_type = "continuous"
                except ValueError:
                    feature_type = "discrete"
            else:
                feature_type = "discrete"

        else:
            raise ValueError("detect_type should be one of 'column_names' or "
                             "'auto'" % detect_column_type)

        return feature_type

    def _get_feature_type(self, feature_name, feature, fit=False):
        """Get the type of a single feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        fit : `bool`
            If `True`, we save the feature type.
            If `False`, we take the corresponding saved feature type.

        Returns
        -------
        output : `str`
            The type of the feature (either `continuous` or `discrete`).
        """
        if fit:
            feature_type = FeaturesBinarizer._detect_feature_type(
                feature, feature_name=feature_name,
                detect_column_type=self.detect_column_type)
            self.feature_type[feature_name] = feature_type

        elif self._fitted:
            feature_type = self.feature_type[feature_name]
        else:
            raise ValueError("cannot call method with fit=True if object "
                             "has not been fitted")

        return feature_type

    @staticmethod
    def _detect_boundaries(feature, n_cuts, method):
        """Boundaries detection of a single feature.

        Parameters
        ----------
        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature

        n_cuts : `int`
            Number of cut points

        method : `str`
            If `quantile`, we use quantiles to construct the intervals.
            If `linspace`, we construct linearly spaced intervals

        Returns
        -------
        output : `np.ndarray`
           The intervals boundaries for the feature.
        """
        if not isinstance(feature.dtype, (int, float)):
            feature = feature.astype(float)

        if method == 'quantile':
            quantile_cuts = np.linspace(0, 100, n_cuts + 2)
            boundaries = np.percentile(feature, quantile_cuts,
                                       interpolation="nearest")
            # Only keep distinct bins boundaries
            boundaries = np.unique(boundaries)
        elif method == 'linspace':
            # Maximum and minimum of the feature
            feat_max = np.max(feature)
            feat_min = np.min(feature)
            # Compute the cuts
            boundaries = np.linspace(feat_min, feat_max, n_cuts + 2)
        else:
            raise ValueError(
                "Method '%s' should be 'quantile' or 'linspace'" % method)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf

        return boundaries

    def _get_boundaries(self, feature_name, feature, fit=False):
        """Get bins boundaries of a single continuous feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature to be binarized

        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(?,)
            The bins boundaries. The number of lines is smaller or
            equal to ``n_cuts``, depending on the ``method`` and/or on
            the actual number of distinct boundaries for this feature.
        """
        if fit:
            if self.method == 'given':
                if self.bins_boundaries is None:
                    raise ValueError("bins_boundaries required when `method` "
                                     "equals 'given'")

                if not isinstance(self.bins_boundaries[feature_name], np.ndarray):
                    raise ValueError("feature %s not found in bins_boundaries" % feature_name)
                boundaries = self.bins_boundaries[feature_name]
            else:
                boundaries = FeaturesBinarizer._detect_boundaries(
                    feature, self.n_cuts, self.method)
                self.bins_boundaries[feature_name] = boundaries
        elif self._fitted:
            boundaries = self.bins_boundaries[feature_name]
        else:
            raise ValueError("cannot call method with fit=True as object has "
                             "not been fit")
        return boundaries

    def _categorical_to_interval(self, feature, feature_name, fit=False):
        """Assign intervals to a single feature considered as `discrete`.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature. Could contain `str` values

        fit : `bool`
            If `True`, we need to fit (compute indexes) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            The discretized feature.
        """
        if fit:
            uniques = np.unique(feature)
            uniques.sort()

            mapper = {
                category: interval
                for interval, category in enumerate(uniques)
            }

            self.mapper[feature_name] = mapper

        else:
            mapper = self.mapper[feature_name]

        def category_to_interval(category):
            if category in mapper:
                return mapper.get(category)
            else:
                return len(list(mapper.keys())) + 1

        return np.vectorize(category_to_interval)(feature)

    def _assign_interval(self, feature_name, feature, fit=False):
        """Assign intervals to a single feature.

        Parameters
        ----------
        feature_name : `str`
            The feature name

        feature : `np.ndarray`, shape=(n_samples,)
            The column containing the feature to be binarized

        fit : `bool`
            If `True`, we need to fit (compute boundaries) for this feature

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples,)
            The discretized feature.
        """
        feature_type = self._get_feature_type(feature_name, feature, fit)

        if feature_type == "continuous":
            if feature.dtype != float:
                feature = feature.astype(float)

            # Get bins boundaries for the feature
            boundaries = self._get_boundaries(feature_name, feature, fit)

            # Discretize feature
            feature = pd.cut(feature, boundaries, labels=False)

        else:
            feature = self._categorical_to_interval(feature, feature_name,
                                                    fit=fit)
        return feature

    def _is_sklearn_older_than(self, ver):
        from packaging import version
        import sklearn
        return version.parse(sklearn.__version__) < version.parse(ver)

    def _get_n_values(self):
        if self._is_sklearn_older_than("0.22.0"):
            return self.one_hot_encoder.n_values_
        else:
            return [len(x) for x in self.one_hot_encoder.categories_]

    def _get_feature_indices(self):
        if self._is_sklearn_older_than("0.22.0"):
            return self.one_hot_encoder.feature_indices_
        else:
            feature_indices = [0]
            for cat in self.one_hot_encoder.categories_:
                feature_indices.append(feature_indices[-1] + len(cat))
            return np.asarray(feature_indices)
