# License: BSD 3 clause

from abc import ABC

from tick.base import Base
from tick.base_model import Model
from tick.prox.base import Prox

from ..build.solver import RandType_perm as perm
from ..build.solver import RandType_unif as unif


class SolverSto(Base):
    """The base class for a stochastic solver.
    In only deals with verbosing information, and setting parameters.

    Parameters
    ----------
    epoch_size : `int`, default=0
        Epoch size. If given before calling set_model, then we'll
        use the specified value. If not, we ``epoch_size`` is specified
        by the model itself, when calling set_model

    rand_type : `str`
        Type of random sampling

        * if ``"unif"`` samples are uniformly drawn among all possibilities
        * if ``"perm"`` a random permutation of all possibilities is
          generated and samples are sequentially taken from it. Once all of
          them have been taken, a new random permutation is generated

    seed : `int`
        The seed of the random sampling. If it is negative then a random seed
        (different at each run) will be chosen.

    Notes
    -----
    This class should not be used by end-users
    """

    _attrinfos = {
        "_solver": {
            "writable": False
        },
        "epoch_size": {
            "writable": True,
            "cpp_setter": "set_epoch_size"
        },
        "_rand_max": {
            "writable": False,
            "cpp_setter": "set_rand_max"
        },
        "_rand_type": {
            "writable": False,
            "cpp_setter": "set_rand_type"
        },
        "seed": {
            "cpp_setter": "set_seed"
        }
    }

    # The name of the attribute that might contain the C++ solver object
    _cpp_obj_name = "_solver"

    def __init__(self, epoch_size: int = None, rand_type: str = "unif",
                 seed=-1):
        Base.__init__(self)
        # The C++ wrapped solver is to be given in child classes
        self._solver = None
        self._rand_type = None
        self._rand_max = None
        self.epoch_size = epoch_size
        self.rand_type = rand_type
        self.seed = seed

    def set_model(self, model: Model):
        # Give the C++ wrapped model to the solver
        self.dtype = model.dtype
        self._solver.set_model(model._model)
        # If not already specified, we use the model's epoch_size
        if self.epoch_size is None:
            self.epoch_size = model._epoch_size
        # We always use the _rand_max given by the model
        self._set_rand_max(model)
        return self

    def set_prox(self, prox: Prox):
        if prox._prox is None:
            raise ValueError("Prox %s is not compatible with stochastic "
                             "solver %s" % (prox.__class__.__name__,
                                            self.__class__.__name__))
            # Give the C++ wrapped prox to the solver
        if self.dtype is None or self.model is None:
            raise ValueError("Solver must call set_model before set_prox")
        if prox.dtype != self.dtype:
            prox = prox.astype(self.dtype)
        self._solver.set_prox(prox._prox)
        return self

    @property
    def rand_type(self):
        if self._rand_type == unif:
            return "unif"
        if self._rand_type == perm:
            return "perm"
        else:
            raise ValueError("No known ``rand_type``")

    @rand_type.setter
    def rand_type(self, val):
        if val not in ["unif", "perm"]:
            raise ValueError("``rand_type`` can be 'unif' or " "'perm'")
        else:
            if val == "unif":
                enum_val = unif
            if val == "perm":
                enum_val = perm
            self._set("_rand_type", enum_val)

    def _set_rand_max(self, model):
        model_rand_max = model._rand_max
        self._set("_rand_max", model_rand_max)

    def _get_typed_class(self, dtype_or_object_with_dtype, dtype_map):
        """Deduce dtype and return true if C++ _model should be set
        """
        import tick.base.dtype_to_cpp_type
        return tick.base.dtype_to_cpp_type.get_typed_class(
            self, dtype_or_object_with_dtype, dtype_map)
