# License: BSD 3 clause

from abc import abstractmethod
from . import Model

__author__ = 'Stephane Gaiffas'


class ModelLipschitz(Model):
    """An abstract base class for a model that implements lipschitz
    constants

    Parameters
    ----------
    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    _attrinfos = {
        "_ready_lip_best": {
            "writable": False
        },
        "_lip_best": {
            "writable": False
        }
    }

    def __init__(self):
        Model.__init__(self)
        self._ready_lip_best = False
        self._lip_best = None

    def fit(self, *args):
        self._set("_ready_lip_best", False)

    def get_lip_max(self) -> float:
        """Returns the maximum Lipschitz constant of individual losses. This is
        particularly useful for step-size tuning of some solvers.

        Returns
        -------
        output : `float`
            The maximum Lipschitz constant
        """
        if self._fitted:
            return self._model.get_lip_max()
        else:
            raise ValueError("call ``fit`` before calling ``get_lip_max``")

    def get_lip_mean(self) -> float:
        """Returns the average Lipschitz constant of individual losses. This is
        particularly useful for step-size tuning of some solvers.

        Returns
        -------
        output : `float`
            The average Lipschitz constant
        """
        if self._fitted:
            return self._model.get_lip_mean()
        else:
            raise ValueError("call ``fit`` before using ``get_lip_max``")

    def get_lip_best(self) -> float:
        """Returns the best Lipschitz constant, using all samples
        Warning: this might take some time, since it requires a SVD computation.

        Returns
        -------
        output : `float`
            The best Lipschitz constant
        """
        if self._fitted:
            if self._ready_lip_best:
                return self._lip_best
            else:
                lip_best = self._get_lip_best()
                self._set("_lip_best", lip_best)
                self._set("_ready_lip_best", True)
                return lip_best
        else:
            raise ValueError("call ``fit`` before calling ``get_lip_best``")

    @abstractmethod
    def _get_lip_best(self) -> float:
        """The method that actually does the computation. Must be overloaded
        in childs
        """
        pass
