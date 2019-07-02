import numpy as np
from operator import itemgetter
from tick.preprocessing.base import LongitudinalPreprocessor


class LongitudinalSamplesFilter(LongitudinalPreprocessor):
    """Longitudinal data preprocessor which filters out samples for which all
    labels are null over the entire observation period.

    Parameters
    ----------
    n_jobs : `int`, default=-1
        Number of tasks to run in parallel. If set to -1, the number of tasks is
        set to the number of cores.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from tick.preprocessing import LongitudinalSamplesFilter
    >>> features = [csr_matrix([[0, 1, 0],
    ...                         [0, 0, 0],
    ...                         [0, 0, 1]], dtype="float64"),
    ...             csr_matrix([[1, 1, 0],
    ...                         [0, 0, 1],
    ...                         [0, 0, 0]], dtype="float64")
    ...             ]
    >>> censoring = np.array([3, 2], dtype="uint64")
    >>> labels = [np.array([0, 1, 0], dtype="uint64"), np.zeros(3, dtype="uint64")]
    >>> n_lags = np.array([2, 1, 0], dtype='uint64')
    >>> lfl = LongitudinalSamplesFilter()
    >>> features, labels, censoring = lfl.fit_transform(features, labels, censoring)
    >>> # output comes as a list of sparse matrices or 2D numpy arrays
    >>> features.__class__
    <class 'list'>
    >>> [x.toarray() for x in features]
   [array([[0., 1., 0.],
           [0., 0., 0.],
           [0., 0., 1.]]), array([[1., 1., 0.],
           [0., 0., 1.],
           [0., 0., 0.]])]
    >>> labels
    [array([0, 1, 0], dtype=uint64), array([0, 0, 0], dtype=uint64)]
    >>> censoring
    array([3, 2], dtype=uint64)
    """

    _attrinfos = {
        "_mask": {
            "writable": False
        },
        "_n_active_patients": {
            "writable": False
        },
        "_n_patients": {
            "writable": False
        },
    }

    def __init__(self, n_jobs=-1):
        LongitudinalPreprocessor.__init__(self, n_jobs=n_jobs)
        self._mask = None
        self._n_active_patients = None
        self._n_patients = None

    def fit(self, features, labels, censoring):
        nnz = [len(np.nonzero(arr)[0]) > 0 for arr in labels]
        self._set('_mask', [
            idx for idx, feat in enumerate(features)
            if feat.sum() > 0 and nnz[idx]
        ])
        self._set('_n_active_patients', len(self._mask))
        self._set('_n_patients', len(features))

        return self

    def transform(self, features, labels, censoring):
        if self._n_active_patients <= 1:
            raise ValueError(
                "There should be more than one positive sample per\
                 batch with nonzero_features. Please check the input data.")
        if self._n_active_patients < self._n_patients:
            features_filter = itemgetter(*self._mask)
            features = features_filter(features)
            labels = features_filter(labels)
            censoring = censoring[self._mask]

        if self._n_active_patients == 1:
            features = [features]
            labels = [labels]
            censoring = np.array(censoring, dtype='uint32')

        return features, labels, censoring
