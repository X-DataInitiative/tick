# License: BSD 3 clause
from tick.optim.proj import ProjHalfSpace
from .build.model import ModelPoisReg as _ModelPoisReg
from .build.model import LinkType_identity as identity
from .build.model import LinkType_exponential as exponential

import numpy as np
from scipy.special import gammaln

from .base import ModelGeneralizedLinear, ModelFirstOrder, ModelSecondOrder, \
    ModelSelfConcordant

__author__ = 'Stephane Gaiffas'


class ModelPoisReg(ModelGeneralizedLinear,
                   ModelSecondOrder,
                   ModelSelfConcordant):
    """Poisson regression model with identity or exponential link for data with
    a count label. This class gives first order and second order information for
    this model (gradient, loss and hessian norm)  and can be passed
    to any solver through the solver's ``set_model`` method.
    Can lead to overflows with some solvers (see the note below).

    Given training data :math:`(x_i, y_i) \\in \\mathbb R^d \\times \\mathbb N`
    for :math:`i=1, \\ldots, n`, this model considers a goodness-of-fit

    .. math::
        f(w, b) = \\frac 1n \\sum_{i=1}^n \\ell(y_i, b + x_i^\\top w),

    where :math:`w \\in \\mathbb R^d` is a vector containing the model-weights,
    :math:`b \\in \\mathbb R` is the intercept (used only whenever
    ``fit_intercept=True``) and
    :math:`\\ell : \\mathbb R^2 \\rightarrow \\mathbb R` is the loss given by

    .. math::
        \\ell(y, y') = e^{y'} - y y'

    whenever ``link='exponential'`` and

    .. math::
        \\ell(y, y') = y' - y \\log(y')

    whenever ``link='identity'``,  for :math:`y \in \mathbb N` and
    :math:`y' \in \mathbb R`. Data is passed
    to this model through the ``fit(X, y)`` method where X is the features
    matrix (dense or sparse) and y is the vector of labels.

    Parameters
    ----------
    fit_intercept : `bool`
        If `True`, the model uses an intercept

    link : `str`, default="exponential"
        Type of link function

        * if ``"identity"``: the intensity is the inner product of the
          model's coeffs with the features. In this case, one
          must ensure that the intensity is non-negative

        * if ``"exponential"``: the intensity is the exponential of the
          inner product of the model's coeffs with the features.

        Note that link cannot be changed after creation of
        `ModelPoisReg`

    Attributes
    ----------
    features : {`numpy.ndarray`, `scipy.sparse.csr_matrix`}, shape=(n_samples, n_features)
        The features matrix, either dense or sparse

    labels : `numpy.ndarray`, shape=(n_samples,) (read-only)
        The labels vector

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    n_threads : `int`, default=1 (read-only)
        Number of threads used for parallel computation.

        * if ``int <= 0``: the number of threads available on
          the CPU
        * otherwise the desired number of threads

    Notes
    -----
    The gradient and loss for the exponential link case cannot be
    overflow proof. In this case, only a solver working in the dual
    (such as `SDCA`) should be used.

    In summary, use grad and call at your own risk when
    ``link="exponential"``
    """

    _attrinfos = {
        "_link_type": {
            "writable": False
        },
        "_link": {
            "writable": False
        }
    }

    def __init__(self, fit_intercept: bool = True,
                 link: str = "exponential", n_threads: int = 1):
        """
        """
        ModelSecondOrder.__init__(self)
        ModelGeneralizedLinear.__init__(self, fit_intercept)
        ModelSelfConcordant.__init__(self)
        self._set("_link", None)
        self.link = link
        self.n_threads = n_threads

    # TODO: implement _set_data and not fit
    def fit(self, features, labels):
        """Set the data into the model object

        Parameters
        ----------
        features : {`numpy.ndarray`, `scipy.sparse.csr_matrix`}, shape=(n_samples, n_features)
            The features matrix, either dense or sparse

        labels : `numpy.ndarray`, shape=(n_samples,)
            The labels vector

        Returns
        -------
        output : `ModelPoisReg`
            The current instance with given data
        """
        ModelFirstOrder.fit(self, features, labels)
        ModelGeneralizedLinear.fit(self, features, labels)
        self._set("_model", _ModelPoisReg(features,
                                          labels,
                                          self._link_type,
                                          self.fit_intercept,
                                          self.n_threads))
        return self

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    @property
    def link(self):
        return self._link

    @link.setter
    def link(self, value):
        if self._link is not None:
            raise ValueError("link is read only")
        if value == "exponential":
            self._set("_link_type", exponential)
        elif value == "identity":
            self._set("_link_type", identity)
        else:
            raise ValueError("``link`` must be either 'exponential' or "
                             "'linear'.")
        self._set("_link", value)

    def _get_sc_constant(self) -> float:
        """Self-concordance parameter of the Poisson
        regression loss
        """
        if self.link == "identity":
            y = self.labels
            return 2 * (1. / np.sqrt(y[y > 0])).max()
        else:
            raise ValueError(("Poisson regression with exponential "
                              "link is not self-concordant"))

    # TODO: C++ for this
    def _hessian_norm(self, coeffs: np.ndarray,
                      point: np.ndarray) -> float:

        link = self.link
        features, labels = self.features, self.labels
        if link == "identity":
            z1 = features.dot(coeffs)
            z2 = features.dot(point)
            # TODO: beware of zeros in z1 or z2 !
            return np.sqrt((labels * z1 ** 2 / z2 ** 2).mean())
        elif link == "exponential":
            raise NotImplementedError("exp link is not yet implemented")
        else:
            raise ValueError("``link`` must be either 'exponential' or "
                             "'linear'.")

    def _sdca_primal_dual_relation(self, l_l2sq, dual_vector):
        # In order to solve the same problem than other solvers, we need to
        # rescale the penalty parameter if some observations are not
        # considered in SDCA. This is useful for Poisson regression with
        # identity link
        if self.link == "identity":
            scaled_l_l2sq = l_l2sq * self.n_samples / self._sdca_rand_max
        else:
            scaled_l_l2sq = l_l2sq

        primal_vector = np.empty(self.n_coeffs)
        self._model.sdca_primal_dual_relation(scaled_l_l2sq, dual_vector,
                                              primal_vector)
        return primal_vector

    @property
    def _sdca_rand_max(self):
        if self.link == "identity":
            non_zero_labels = self.labels != 0
            return non_zero_labels.sum().item()
        else:
            raise NotImplementedError()

    def dual_loss(self, dual_coeffs):
        """Computes the dual loss at the given dual coefficients

        Parameters
        ----------
        dual_coeffs : `np.ndarray`
            Dual coefficients

        Returns
        -------
        dual_loss : `float`
            The value of the dual loss
        """
        if self.link != "identity":
            raise (NotImplementedError())

        non_zero_labels = self.labels != 0
        dual_loss = self.labels[non_zero_labels] * (
            1 + np.log(dual_coeffs / self.labels[non_zero_labels]))
        dual_loss += np.mean(gammaln(self.labels[non_zero_labels] + 1))

        return np.mean(dual_loss) * self._sdca_rand_max / self.n_samples

    def hessian(self, x):
        """Return model's hessian

        Parameters
        ----------
        x : `np.ndarray`, shape=(n_coeffs,)
            Value at which the hessian is computed
        """
        return self._model.hessian(x)

    def create_proj(self):
        A = self.features
        A = A[self.labels > 0, :]
        if self.fit_intercept:
            intercept = np.ones(len(A)).reshape(len(A), 1)
            A = np.hstack((A, intercept))
            A = np.ascontiguousarray(A.astype(float))

        b = 1e-8 + np.zeros(A.shape[0])
        return ProjHalfSpace(max_iter=1000).fit(A, b)

    def get_dual_bounds(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)
        feature_norms = np.linalg.norm(features, axis=1)
        non_zero_features = features[labels != 0]
        n_psi_x = n_non_zeros * np.mean(features, axis=0)\
            .dot(non_zero_features.T)
        tmp = np.power(n_psi_x, 2)
        tmp += 4 * l_l2sq * n_non_zeros * labels[labels != 0] * feature_norms

        return 0.5 / feature_norms * (n_psi_x + np.sqrt(tmp))

    def get_dual_init(self, l_l2sq, init_type='psi'):

        if init_type == 'psi':
            kappa = self._get_unscaled_dual_init(l_l2sq)
        elif init_type == 'features':
            kappa = self._get_unscaled_dual_init_features(l_l2sq)
        elif init_type == 'sqrt':
            kappa = self._get_unscaled_dual_init_sqrt(l_l2sq)
        elif init_type == 'log':
            kappa = self._get_unscaled_dual_init_log(l_l2sq)
        else:
            raise ValueError('Unknown init type {}'.format(init_type))

        dual_inits = self._get_dual_init_estimated_base(l_l2sq, kappa) * kappa

        return dual_inits

    def _get_unscaled_dual_full_init(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)
        non_zero_features = features[labels != 0]
        non_zero_labels = labels[labels != 0]

        features_sum = np.sum(non_zero_features, axis=0)
        features_dot_features_sum = non_zero_features.dot(features_sum)
        n_psi = np.sum(features, axis=0)
        n_psi_x = n_psi.dot(non_zero_features.T)

        tmp1 = np.power(n_psi_x, 2)
        tmp1 += 4 * l_l2sq * n_non_zeros * non_zero_labels * \
                features_dot_features_sum

        inits = n_psi_x + np.sqrt(tmp1)
        inits /= (2 * features_dot_features_sum)
        return inits

    def _get_unscaled_dual_init(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)
        non_zero_features = features[labels != 0]
        non_zero_labels = labels[labels != 0]

        features_sum = np.sum(non_zero_features, axis=0)
        features_dot_features_sum = non_zero_features.dot(features_sum)
        n_psi = np.sum(features, axis=0)
        n_psi_x = n_psi.dot(non_zero_features.T)

        tmp1 = np.power(n_psi_x, 2)
        tmp1 += 4 * l_l2sq * n_non_zeros * non_zero_labels * \
               features_dot_features_sum
        inits1 = n_psi_x + np.sqrt(tmp1)
        inits1 /= (2 * features_dot_features_sum)

        # tmp2 = features_dot_features_sum /

        # print('l2 n', l_l2sq * n_non_zeros, 'tmp - 1', tmp)
        # inits -= 1

        # corr = l_l2sq * n_non_zeros * non_zero_labels
        # corr *= features_dot_features_sum / np.power(n_psi_x, 2)
        corr = n_non_zeros * non_zero_labels / n_psi_x

        return corr

    def _get_unscaled_dual_init_sqrt(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)
        non_zero_features = features[labels != 0]
        non_zero_labels = labels[labels != 0]

        n_psi = np.sum(features, axis=0)
        n_psi_x = n_psi.dot(non_zero_features.T)

        corr = n_non_zeros * np.sqrt(non_zero_labels) / n_psi_x

        return corr


    def _get_unscaled_dual_init_log(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)
        non_zero_features = features[labels != 0]
        non_zero_labels = labels[labels != 0]

        n_psi = np.sum(features, axis=0)
        n_psi_x = n_psi.dot(non_zero_features.T)

        corr = n_non_zeros * np.log(1 + non_zero_labels) / n_psi_x

        return corr

    def _get_unscaled_dual_init_features(self, l_l2sq):
        labels = self.labels
        features = self.features

        n_non_zeros = sum(labels != 0)

        non_zero_features = features[labels != 0]
        non_zero_labels = labels[labels != 0]

        features_sum = np.sum(non_zero_features, axis=0)

        features_dot_x = features_sum.dot(non_zero_features.T)

        corr = n_non_zeros * non_zero_labels / features_dot_x

        return corr

    def _get_dual_init_estimated_base(self, l_l2sq, kappa_i):
        n_non_zeros = sum(self.labels != 0)
        non_zero_features = self.features[self.labels != 0]

        phi = np.sum((kappa_i * non_zero_features.T), axis=1)
        psi = np.sum(self.features, axis=0) / n_non_zeros

        if self.fit_intercept:
            phi = np.hstack((phi, np.sum(kappa_i)))
            psi = np.hstack((psi, self.n_samples / n_non_zeros))

        norm_phi_sq = np.linalg.norm(phi) ** 2
        n_phi_dot_psi = n_non_zeros * psi.dot(phi)

        tmp = (n_phi_dot_psi ** 2 +
               4 * l_l2sq * n_non_zeros * self.labels.sum() * norm_phi_sq)

        return 0.5 / norm_phi_sq * (n_phi_dot_psi + np.sqrt(tmp))
