import numpy as np
from scipy.stats import norm


class BootstrapRelativeRisksMetrics:
    """Provides confidence intervals, boostrap mean and boostrap stdev which can then
    be used to compute approximate p-values and power.
    See All of Statistics (Wasserman, 2003), pp. 134-135 for the justification.

    We compute the metrics on relative risks rather than raw coeffs (log space) to
    avoid loosing precision, as coeffs can be very close to zero, while risks are
    between .5 and 50 most of the time.

    All the metrics are approximates resulting from bootstrap estimates.

    Parameters
    __________
    model_coefficients : `list`
        List containing 1-dimensional `np.ndarray` (`dtype=float`)
        containing the coefficients of the model used to compute bootstrap samples.
        Each numpy array contains the `(n_lags + 1)` coefficients associated with a
        feature. Each coefficient of such arrays can be interpreted as the log relative
        intensity associated with this feature, `k` periods after exposure
        start, where `k` is the index of the coefficient in the array.

    bootstrap_samples : `list`
        List containings coefficients (in a format similar to `model_coefficients`)
        resulting from each bootstrap iteration.

    confidence: `float`
        Confidence level used for the computation of approximate confidence intervals.


    Attributes
    __________
    relative_risk: `numpy.ndarray`
        Relative risks (exponential of the model coefficients).

    bootstrap_relative_risk_std: `numpy.ndarray`
        Standard deviation of the relative risks estimated approximated using boostrap.

    relative_risk_ci: `numpy.ndarray`
        Relative risks confidence intervals approximated using boostrap
        (quantile method). The confidence interval might not be symmetric when using
        too few boostrap samples.

    relative_risk_ci_symmetric: `numpy.ndarray`
        Relative risks confidence intervals approximated using
        `bootstrap_relative_risk_std`.

    p_values: `numpy.ndarray`
        P-values for individual wald tests testing if risk is significantly different
        from the neutral risk (=1). These p-values are computed using
        `bootstrap_relative_risk_std`.

    power: `numpy.ndarray`
        Approximated power at `confidence` level, these can be used to put have a
        better understanding of confidence intervals.

    power_at_pval: `numpy.ndarray`
        Approximated power at `p_values` levels.
    """

    def __init__(self, model_coefficients, bootstrap_samples, confidence):
        self.confidence = confidence
        offset = [0]
        offset.extend([len(c) for c in model_coefficients])
        self._features_offset = np.cumsum(np.array(offset))

        bootstrap_samples_ = np.array(bootstrap_samples)
        bootstrap_relative_risk_samples = np.exp(bootstrap_samples_)
        self.relative_risk = np.exp(np.array(model_coefficients).ravel())
        self.bootstrap_relative_risk_std = np.std(bootstrap_relative_risk_samples,
                                                  axis=0)
        self._z_alpha = norm.ppf(1 - (confidence / 2))

        # CI with quantile method
        relative_risk_lower_bound = \
            self.get_bootstrap_coeffs_quantile(bootstrap_relative_risk_samples,
                                               self.confidence / 2)
        relative_risk_upper_bound = \
            self.get_bootstrap_coeffs_quantile(bootstrap_relative_risk_samples,
                                               1 - self.confidence / 2)
        self.relative_risk_ci = (relative_risk_lower_bound, relative_risk_upper_bound)

        # Symmetric CI
        relative_risk_lower_bound_s = self.relative_risk - \
            self.bootstrap_relative_risk_std * self._z_alpha
        relative_risk_upper_bound_s = self.relative_risk + \
            self.bootstrap_relative_risk_std * self._z_alpha
        self.relative_risk_ci_symmetric = (relative_risk_lower_bound_s,
                                           relative_risk_upper_bound_s)

        neutral_risk_hypothesis = np.ones_like(self.relative_risk)
        self.p_values = self.compute_approximate_wald_p_values(
            neutral_risk_hypothesis,
            self.relative_risk,
            self.bootstrap_relative_risk_std,
        )
        self.power = self.compute_approximate_wald_power(
            neutral_risk_hypothesis,
            self.relative_risk,
            self.bootstrap_relative_risk_std,
            self.confidence
        )
        self.power_at_pval = self.compute_approximate_wald_power(
            neutral_risk_hypothesis,
            self.relative_risk,
            self.bootstrap_relative_risk_std,
            self.p_values
        )

    def _asdict(self):
        output = {
            'relative_risks': self._format(self.relative_risk),
            'ci': (self._format(self.relative_risk_ci[0]),
                   self._format(self.relative_risk_ci[1])),
            'symmetric_ci': (self._format(self.relative_risk_ci_symmetric[0]),
                             self._format(self.relative_risk_ci_symmetric[1])),
            'p-values': self._format(self.p_values),
            'power': self._format(self.power),
            'power_at_p-value': self._format(self.power_at_pval),
            'estimates_stdev': self._format(self.bootstrap_relative_risk_std)
        }
        return output

    def _format(self, coeffs):
        value = list()
        for i, o in enumerate(self._features_offset[:-1]):
            start = int(o)
            end = int(self._features_offset[i+1])
            value.append(coeffs[start:end])
        value = np.array(value).tolist()
        return value

    @staticmethod
    def get_bootstrap_coeffs_quantile(bootstrap_samples, quantile):
        """Approximate confidence intervals computed using parametric bootstrap and
        bootstrap percentile intervals method, see  All of Statistics (Wasserman, 2003),
        p. 111."""
        rep = bootstrap_samples.shape[0]
        samples = np.sort(bootstrap_samples, 0)
        return samples[int(np.floor(rep * quantile))]

    @staticmethod
    def compute_approximate_wald_p_values(hypothesis, estimates, estimates_std):
        """Compute p_values for Wald test:
        H_0 : estimates == hypothesis
        H_1 : estimates != hypothesis
        where estimates_std is the standard deviation of the estimated coefficients.

        Approximate p-values, see All of Statistics (Wasserman, 2003), p. 158.
        """
        wald_statistics = (estimates - hypothesis) / estimates_std
        return 2 * norm.cdf(-np.abs(wald_statistics))

    @staticmethod
    def compute_approximate_wald_power(hypothesis, estimates, estimates_std, confidence):
        """Compute approximate power for Wald test:
        H_0 : estimates == hypothesis
        H_1 : estimates != hypothesis
        where estimates_std is the standard deviation of the estimated coefficients.

        Approximate power, see All of Statistics (Wasserman, 2003), p. 153.
        """
        z_alpha = norm.ppf(1 - (confidence / 2))
        wald_statistics = (estimates - hypothesis) / estimates_std
        return 1 - norm.cdf(wald_statistics + z_alpha) + norm.cdf(wald_statistics - z_alpha)
