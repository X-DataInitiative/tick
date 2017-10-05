# License: BSD 3 clause

import numpy as np
from scipy.special import gammaln

from tick.optim.model import ModelPoisReg
from tick.optim.model.base import ModelGeneralizedLinear, ModelFirstOrder


class ModelPoisRegSDCA(ModelFirstOrder, ModelGeneralizedLinear):

    _attrinfos = {
        '_poisreg': {}
    }

    def __init__(self, l_l2sq, fit_intercept: bool = True):
        """
        """
        ModelFirstOrder.__init__(self)
        ModelGeneralizedLinear.__init__(self, fit_intercept)
        self.l_l2sq = l_l2sq
        self._poisreg = ModelPoisReg(fit_intercept, link='identity')

    # TODO: implement _set_data and not fit
    def fit(self, features, labels):
        """Set the data into the model object

        Parameters
        ----------
        features : `np.ndarray`, shape=(n_samples, n_features)
            The features matrix

        labels : `np.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : `ModelPoisReg`
            The current instance with given data
        """
        self._poisreg.fit(features, labels)
        ModelGeneralizedLinear.fit(self, features, labels)
        return self

    def _grad(self, dual_vector: np.ndarray, out: np.ndarray) -> None:
        labels = self.labels
        features = self.features
        n_samples = len(labels)
        non_zero_features = features[labels != 0]

        alpha_x = np.sum(np.diag(dual_vector).dot(non_zero_features),
                         axis=0)
        psi_x = np.sum(features, axis=0).dot(non_zero_features.T)

        out[:] = 1. / n_samples * labels[labels != 0] / dual_vector
        out -= 1. / (self.l_l2sq * n_samples ** 2) * alpha_x.dot(
            non_zero_features.T)
        out += 1. / (self.l_l2sq * n_samples ** 2) * psi_x
        out *= -1
        # print(out)

    def _loss(self, coeffs: np.ndarray) -> float:
        primal = self.get_primal(coeffs)
        prox_l2_value = 0.5 * self.l_l2sq * np.linalg.norm(primal) ** 2
        loss = self._poisreg.dual_loss(coeffs) - prox_l2_value
        loss *= -1
        return loss

    def get_primal(self, dual_vector):
        return self._poisreg._sdca_primal_dual_relation(
            self.l_l2sq, dual_vector)

    def _get_n_coeffs(self) -> int:
        return sum(self.labels != 0)