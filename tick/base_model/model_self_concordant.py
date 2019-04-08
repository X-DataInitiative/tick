# License: BSD 3 clause

from abc import abstractmethod
from . import Model

__author__ = 'Stephane Gaiffas'


class ModelSelfConcordant(Model):
    """An abstract base class for a model that implements the
    self-concordant constant

    Attributes
    ----------
    dtype : `{'float64', 'float32'}`, default='float64'
        Type of the arrays used. This value is set from model and prox dtypes.

    Notes
    -----
    This class should be not used by end-users, it is intended for
    development only.
    """

    def __init__(self):
        Model.__init__(self)

    @property
    def _sc_constant(self) -> float:
        if not self._fitted:
            raise ValueError("call ``fit`` before using " "``sc_constant``")
        return self._get_sc_constant()

    @abstractmethod
    def _get_sc_constant(self) -> float:
        pass
