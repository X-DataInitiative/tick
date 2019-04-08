# License: BSD 3 clause

from tick.base import actual_kwargs
from .base import LearnerRobustGLM
from .model_linreg_with_intercepts import ModelLinRegWithIntercepts


class RobustLinearRegression(LearnerRobustGLM):
    """Robust linear regression learner.
    This is linear regression with sample intercepts (one for each sample).
    An ordered-L1 penalization is used on the individual intercepts, in order to
    obtain a sparse vector of individual intercepts, with a (theoretically)
    guaranteed False Discovery Rate control (see ``fdr`` below).

    The features matrix should contain only continuous features, and columns
    should be normalized.
    Note that C_sample_intercepts is a sensitive parameter, that should be
    tuned in theory as n_samples / noise_level, where noise_level can be chosen
    as a robust estimation of the standard deviation, using for instance
    `tick.hawkes.inference.std_mad` and `tick.hawkes.inference.std_iqr` on the array of
    labels.

    Parameters
    ----------
    C_sample_intercepts : `float`
        Level of penalization of the ProxSlope penalization used for detection
        of outliers. Ideally, this should be equal to n_samples / noise_level,
        where noise_level is an estimated noise level

    C : `float`, default=1e3
        Level of penalization of the model weights

    fdr : `float`, default=0.05
        Target false discovery rate for the detection of outliers, namely for
        the detection of non-zero entries in ``sample_intercepts``

    penalty : {'none', 'l1', 'l2', 'elasticnet', 'slope'}, default='l2'
        The penalization to use. Default 'l2', namely ridge penalization.

    fit_intercept : `bool`, default=True
        If `True`, include an intercept in the model, namely a global intercept

    refit : 'bool', default=False
        Not implemented yet

    solver : 'gd', 'agd'
        The name of the solver to use. For now, only gradient descent and
        accelerated gradient descent are available

    warm_start : `bool`, default=False
        If true, learning will start from the last reached solution

    step : `float`, default=None
        Initial step size used for learning.

    tol : `float`, default=1e-7
        The tolerance of the solver (iterations stop when the stopping
        criterion is below it).

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
        Used in 'elasticnet' penalty only.

    slope_fdr : `float`, default=0.05
        Target false discovery rate for the detection of detection of non-zero
        entries in the model weights.
        Used in the 'slope' penalty only.

    Attributes
    ----------
    weights : `numpy.array`, shape=(n_features,)
        The learned weights of the model (not including the intercept)

    sample_intercepts : `numpy.array`, shape=(n_samples,)
        Sample intercepts. This should be a sparse vector, since a non-zero
        entry means that the sample is an outlier.

    intercept : `float` or `None`
        The intercept, if ``fit_intercept=True``, otherwise `None`

    coeffs : `numpy.array`, shape=(n_features + n_samples + 1,)
        The full array of coefficients of the model. Namely, this is simply
        the concatenation of ``weights``, ``sample_intercepts``
        and ``intercept``
    """

    _attrinfos = {"_actual_kwargs": {"writable": False}}

    @actual_kwargs
    def __init__(self, C_sample_intercepts, C=1e3, fdr=0.05, penalty='l2',
                 fit_intercept=True, refit=False, solver='agd',
                 warm_start=False, step=None, tol=1e-7, max_iter=200,
                 verbose=True, print_every=10, record_every=10,
                 elastic_net_ratio=0.95, slope_fdr=0.05):
        self._actual_kwargs = RobustLinearRegression.__init__.actual_kwargs
        LearnerRobustGLM.__init__(
            self, C_sample_intercepts=C_sample_intercepts, C=C, fdr=fdr,
            penalty=penalty, fit_intercept=fit_intercept, refit=refit,
            solver=solver, warm_start=warm_start, step=step, tol=tol,
            max_iter=max_iter, verbose=verbose, print_every=print_every,
            record_every=record_every, elastic_net_ratio=elastic_net_ratio,
            slope_fdr=slope_fdr)

    def _construct_model_obj(self, fit_intercept=True):
        return ModelLinRegWithIntercepts(fit_intercept)
