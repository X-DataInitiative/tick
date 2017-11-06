__author__ = 'stephanegaiffas'


# TODO: projection onto the half-space
# TODO: projection onto an intersection of half-spaces
# TODO: projection onto the simplex


from abc import ABCMeta, abstractmethod
import numpy as np
from tick.base import Base
from .build.proj import proj_simplex, proj_half_spaces


class Proj():
    """An abstract base class for a projection operator
    """
    __metaclass__ = ABCMeta

    def call(self, coeffs: np.ndarray, t: float = 1.,
             out: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        coeffs : np.ndarray, shape=(n_params,)
            Input vector
        t : float, default=1.
            This is useless for projection operators, but kept to respect the interface of Prox
        out : np.ndarray, shape=(n_params,), default=None
            If not None, the output of the proximal operator is stored in this vector.
            Otherwise, a new vector is created.
        """
        if out is None:
            # We don't have an output vector, we create a fresh copy
            out = coeffs.copy()
        else:
            # We do an inplace copy of coeffs into out
            out[:] = coeffs
        # Apply the proximal, the output is in out
        self._call(coeffs, out)
        return out

    @abstractmethod
    def _call(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        """Computes the gradient at ``coeffs``. The output must be
        saved in out.

        Notes
        -----
        Must be overloaded in child class
        """
        pass

    def value(self, coeffs: np.ndarray) -> float:
        """
        Notes
        -----
        The value of a projection is always 0.
        """
        return 0.


class ProjSimplex(Proj):
    """Projection onto the simplex.

    References
    ----------
    Insert references here
    """

    def __init__(self, **kwargs):
        """
        Keyword arguments
        -----------------
        l_simplex: float, default=1.
            Radius of the simplex.
        """
        super().__init__(**kwargs)
        self.set_params(l_simplex=1.)
        self.set_params(**kwargs)

    def _call(self, coeffs: np.ndarray, out: np.ndarray):
        r = self.get_params("l_simplex")
        # out always contains a copy of coeffs
        # Sort out in reverse order put it again in out
        # TODO: do the sorting in C++ instead ?
        out[:] = -np.sort(-out)
        # Call the C++ routine
        proj_simplex(coeffs, out, r)


class ProjHalfSpace(Proj):
    """Projection onto a half-space, or an intersection of half-spaces, namely onto the set

    C = { x \in \mathbb R^d \; : \; A x >= b }
    """

    def __init__(self, max_iter=50):
        super().__init__()
        self.max_iter = max_iter

    def fit(self, A: np.ndarray, b: np.ndarray):
        """Give the"""
        # TODO: check that the intersection of half spaces is non-empty
        n_constraints, n_coeffs = A.shape
        self.n_constraints = n_constraints
        self.n_coeffs = n_coeffs
        # Compute the Euclidian norms of the lines of A
        self.A = A
        self.b = b
        # Precompute norms once for all
        self._norms = np.linalg.norm(A, axis=1) ** 2
        max_iter = self.max_iter
        self.history = np.empty(self.max_iter, dtype=np.double)
        return self

    def _call(self, coeffs: np.ndarray, out: np.ndarray):
        # Call the C++ routine
        n_pass = proj_half_spaces(coeffs, self.A, self.b, self._norms, out,
                                  self.max_iter, self.history)
        self.n_pass = n_pass
