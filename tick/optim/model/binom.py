from .base import ModelFirstOrder
import numpy as np

__author__ = 'stephanegaiffas'


class ModelBinomOutlier(ModelFirstOrder):
    def __init__(self, **kwargs):
        """
        """
        ModelFirstOrder.__init__(self, **kwargs)

    def _get_n_coeffs(self, *args, **kwargs):
        """Number of parameters is 1 + n_samples + n_features
        """
        return self.get_info("n_samples") + self.get_info("n_features") + 1

    def _set_data(self, features: np.ndarray, labels: np.ndarray, tosses: np.ndarray, *args, **kwargs):
        """Set the data into the gradient object

        :param features: ndarray, shape=[n_samples, n_features]
            the features matrix

        :param labels: ndarray, shape=[n_samples,]
            the labels vector, valued in N

        """
        n_samples, n_features = features.shape
        self.features = features.reshape(n_samples, n_features)
        self.labels = labels.reshape(n_samples, 1)
        self.tosses = tosses.reshape(n_samples, 1)
        self._set_info(n_samples=n_samples, n_features=n_features)
        return self

    def __get_params(self, coeffs: np.ndarray) -> tuple:
        """We store the parameters in the following order: bias, intercept and beta

        :param coeffs:
        :return:
        """
        n_samples = self.get_info("n_samples")
        n_features = self.get_info("n_features")
        return coeffs[0], coeffs[1:n_samples + 1].reshape(n_samples, 1), \
               coeffs[n_samples + 1:].reshape(n_features, 1)

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Computes the gradient at coeffs.
        Must be overloaded in child class
        """
        if out is None:
            out = self.__grad

        n_samples = self.get_info("n_samples")
        labels = self.labels
        tosses = self.tosses
        features = self.features
        bias, intercept, beta = self.__get_params(coeffs)
        prob = (labels / tosses).reshape(n_samples, 1)
        z = bias + intercept + features.dot(beta)
        # TODO: overflow proof logistic
        s = 1 / (1 + np.exp(-z))
        delta = tosses * (s - prob).reshape(n_samples, 1)
        # Gradient with respect to the bias
        out[0] = delta.mean()
        # Gradient with respect to the intercept
        out[1:n_samples + 1] = delta.reshape(n_samples, ) / n_samples
        # Gradient with respect to the beta
        out[n_samples + 1:] = (delta * features).mean(axis=0)
        return out

    def _loss(self, coeffs):
        """Computes the value of the function at coeffs, for which we compute
        the gradient

        Must be overloaded in child class
        """
        n_samples = self.get_info("n_samples")
        labels = self.labels
        tosses = self.tosses
        features = self.features
        bias, intercept, beta = self.__get_params(coeffs)
        prob = (labels / tosses).reshape(n_samples, 1)
        z = bias + intercept + features.dot(beta)
        # TODO: overflow proof logistic
        return -(tosses * (prob * z - np.log(1 + np.exp(z)))).mean()
