# License: BSD 3 clause

# -*- coding: utf8 -*-

import numpy as np
from numpy.linalg import svd
from sklearn.utils.extmath import randomized_svd

from tick.prox.base import Prox

__author__ = 'Stephane Gaiffas'

# TODO: code the incremental strategy, where we try smaller SVDs

class ProxNuclear(Prox):
    """Proximal operator of the nuclear norm, aka trace norm

    Parameters
    ----------
    strength : `float`
        Level of penalization

    n_rows : `int`
        Number of rows in the matrix on which we apply this
        penalization. The number of columns is then given by
        (start - end) / n_rows

    range : `tuple` of two `int`, default=`None`
        Range on which the prox is applied. If `None` then the prox is
        applied on the whole vector

    positive : `bool`, default=`False`
        If True, apply nuclear-norm penalization followed by a
        truncation to make all entries non-negative

    rank_max : `int`, default=`None`
        Maximum rank to be used in the SVD (not used yet...)

    Notes
    -----
    The coeffs on which we apply this prox must be flattened (using
    `np.ravel` for instance), and not two-dimensional.
    This operator is not usable from a solver with wrapped C++ code.
    It is based on `scipy.linalg.svd` SVD routine and is not intended
    for use on large matrices
    """
    _N_ITER_SVD = 5

    _attrinfos = {
        '_n_components': {}
    }

    def __init__(self, strength: float, n_rows: int = None,
                 range: tuple = None, positive: bool = False):
        Prox.__init__(self, range)
        self.positive = positive
        self.strength = strength
        self.n_rows = n_rows
        self.rank_max = None
        self._n_components = None

    def _get_matrix(self, coeffs):
        if self.n_rows is None:
            raise ValueError("'n_rows' parameter must be set before, either "
                             "in constructor or manually")

        range = self.range
        if range is None:
            start, end = 0, coeffs.shape[0]
        else:
            start, end = range
        n_rows = self.n_rows
        if (end - start) % n_rows:
            raise ValueError("``end``-``start`` must be a multiple of "
                             "``n_rows``")
        n_cols = int((end - start) / n_rows)
        x = coeffs[start:end].copy().reshape((n_rows, n_cols))
        return x

    def _perform_svd(self, x, thresh):
        max_components = min(x.shape)
        if self._n_components is None:
            n_components = max_components
        else:
            n_components = min(self._n_components, max_components)

        u, s, v = randomized_svd(x, n_components,
                                 n_iter=ProxNuclear._N_ITER_SVD)
        if s.min() >= thresh and n_components < max_components:
            # We didn't try enough components
            self._n_components = 1 + int(len(s) * 1.5)
            # retry with more components
            # print('need to retry (tried {}/{})'
            #       .format(n_components, max_components))
            return self._perform_svd(x, thresh)
        elif s.min() >= thresh and n_components == max_components:
            # print('worked with {}/{} (ie. full)'
            #       .format(n_components, max_components))
            self._n_components = max_components
        else:
            # print('worked with {}/{}'.format(n_components, max_components))
            needed_components = np.argmax((s - thresh) < 0)
            # We add little extra to avoid to retry to often
            extra = max(1, int(0.1 * needed_components))
            self._n_components = needed_components + extra

        # print(s - thresh)
        return u, s, v

    def _call(self, coeffs: np.ndarray, step: float, out: np.ndarray):
        x = self._get_matrix(coeffs)
        thresh = step * self.strength
        u, s, v = self._perform_svd(x, thresh)

        s = (s - thresh) * (s > thresh)
        x_new = u.dot(np.diag(s)).dot(v).ravel()
        if self.positive:
            x_new[x_new < 0.] = 0.
        if self.range is None:
            start, end = 0, coeffs.shape[0]
        else:
            start, end = self.range
        out[start:end] = x_new

    def value(self, coeffs: np.ndarray):
        """
        Returns the value of the penalization at ``coeffs``

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape=(n_coeffs,)
            The value of the penalization is computed at this point

        Returns
        -------
        output : `float`
            Value of the penalization at ``coeffs``
        """
        x = self._get_matrix(coeffs)
        # if x.shape[0] != x.shape[1]:
        #     raise ValueError('Prox nuclear must be called on a squared matrix'
        #                      ', received {} np.ndarray'.format(x.shape))
        _, s, _ = randomized_svd(x, min(x.shape),
                                 n_iter=ProxNuclear._N_ITER_SVD)
        # print('full value')
        return self.strength * s.sum()
