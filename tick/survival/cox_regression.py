# License: BSD 3 clause

import numpy as np

from tick.base import actual_kwargs
from tick.preprocessing.utils import safe_array
from .model_coxreg_partial_lik import ModelCoxRegPartialLik
from tick.base.learner import LearnerOptim


class CoxRegression(LearnerOptim):
    """Cox regression learner, using the partial Cox likelihood for
    proportional risks, with many choices of penalization.

    Note that this learner does not have predict functions

    Parameters
    ----------
    C : `float`, default=1e3
        Level of penalization

    penalty : {'none', 'l1', 'l2', 'elasticnet', 'tv', 'binarsity'}, default='l2'
        The penalization to use. Default is 'l2', namely Ridge penalization

    solver : {'gd', 'agd'}, default='agd'
        The name of the solver to use.

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning. Used when solver is 'gd' or
        'agd'.

    tol : `float`, default=1e-5
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it). By default the solver does ``max_iter``
        iterations

    max_iter : `int`, default=100
        Maximum number of iterations of the solver

    verbose : `bool`, default=True
        If `True`, we verbose things, otherwise the solver does not
        print anything (but records information in history anyway)

    print_every : `int`, default=10
        Print history information when ``n_iter`` (iteration number) is
        a multiple of ``print_every``

    record_every : `int`, default=10
        Record history information when ``n_iter`` (iteration number) is
        a multiple of ``record_every``

    Other Parameters
    ----------------
    elastic_net_ratio : `float`, default=0.95
        Ratio of elastic net mixing parameter with 0 <= ratio <= 1.
        For ratio = 0 this is ridge (L2 squared) regularization
        For ratio = 1 this is lasso (L1) regularization
        For 0 < ratio < 1, the regularization is a linear combination
        of L1 and L2.
        Used in 'elasticnet' penalty

    random_state : int seed, RandomState instance, or None (default)
        The seed that will be used by stochastic solvers. Used in 'sgd',
        'svrg', and 'sdca' solvers

    blocks_start : `numpy.array`, shape=(n_features,), default=None
        The indices of the first column of each binarized feature blocks. It
        corresponds to the ``feature_indices`` property of the
        ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    blocks_length : `numpy.array`, shape=(n_features,), default=None
        The length of each binarized feature blocks. It corresponds to the
        ``n_values`` property of the ``FeaturesBinarizer`` preprocessing.
        Used in 'binarsity' penalty

    Attributes
    ----------
    coeffs : np.array, shape=(n_features,)
        The learned coefficients of the model
    """

    _solvers = {'gd': 'GD', 'agd': 'AGD'}

    _attrinfos = {"_actual_kwargs": {"writable": False}}

    @actual_kwargs
    def __init__(self, penalty='l2', C=1e3, solver='agd', step=None, tol=1e-5,
                 max_iter=100, verbose=False, warm_start=False, print_every=10,
                 record_every=10, elastic_net_ratio=0.95, random_state=None,
                 blocks_start=None, blocks_length=None):

        self._actual_kwargs = CoxRegression.__init__.actual_kwargs
        LearnerOptim.__init__(
            self, penalty=penalty, C=C, solver=solver, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, warm_start=warm_start,
            print_every=print_every, record_every=record_every,
            sdca_ridge_strength=0, elastic_net_ratio=elastic_net_ratio,
            random_state=random_state, blocks_start=blocks_start,
            blocks_length=blocks_length)
        self.coeffs = None

    def _construct_model_obj(self):
        return ModelCoxRegPartialLik()

    def _all_safe(self, features: np.ndarray, times: np.array,
                  censoring: np.array):
        if not set(np.unique(censoring)).issubset({0, 1}):
            raise ValueError('``censoring`` must only have values in {0, 1}')
        # All times must be positive
        if not np.all(times >= 0):
            raise ValueError('``times`` array must contain only non-negative '
                             'entries')
        features = safe_array(features)
        times = safe_array(times)
        censoring = safe_array(censoring, np.ushort)
        return features, times, censoring

    def fit(self, features: np.ndarray, times: np.array, censoring: np.array):
        """Fit the model according to the given training data.

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `CoxRegression`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below

        solver_obj = self._solver_obj
        model_obj = self._model_obj
        prox_obj = self._prox_obj

        features, times, censoring = self._all_safe(features, times, censoring)

        # Pass the data to the model
        model_obj.fit(features, times, censoring)

        if self.step is None and self.solver in self._solvers_with_step:
            if self.solver in self._solvers_with_linesearch:
                self._solver_obj.linesearch = True

        # No intercept in this model
        prox_obj.range = (0, model_obj.n_coeffs)

        # Now, we can pass the model and prox objects to the solver
        solver_obj.set_model(model_obj).set_prox(prox_obj)

        coeffs_start = None
        if self.warm_start and self.coeffs is not None:
            coeffs = self.coeffs
            # ensure starting point has the right format
            if coeffs.shape == (model_obj.n_coeffs,):
                coeffs_start = coeffs

        # Launch the solver
        coeffs = solver_obj.solve(coeffs_start)

        # Get the learned coefficients
        self._set("coeffs", coeffs)
        self._set("_fitted", True)
        return self

    def score(self, features=None, times=None, censoring=None):
        """Returns the negative log-likelihood of the model, using the current
        fitted coefficients on the passed data.
        If no data is passed, the negative log-likelihood is computed using the
        data used for training.

        Parameters
        ----------
        features : `None` or `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `None` or `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `None` or `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `float`
            The value of the negative log-likelihood
        """
        if self._fitted:
            all_none = all(e is None for e in [features, times, censoring])
            if all_none:
                return self._model_obj.loss(self.coeffs)
            else:
                if features is None:
                    raise ValueError('Passed ``features`` is None')
                elif times is None:
                    raise ValueError('Passed ``times`` is None')
                elif censoring is None:
                    raise ValueError('Passed ``censoring`` is None')
                else:
                    features, times, censoring = self._all_safe(
                        features, times, censoring)
                    model = ModelCoxRegPartialLik().fit(
                        features, times, censoring)
                    return model.loss(self.coeffs)
        else:
            raise RuntimeError('You must fit the model first')
