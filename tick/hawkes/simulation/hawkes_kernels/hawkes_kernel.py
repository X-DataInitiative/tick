# License: BSD 3 clause

from tick.base import Base
from tick.hawkes.simulation.build.hawkes_simulation import (HawkesKernel as
                                                            _HawkesKernel)


class HawkesKernel(Base):
    """ The kernel class allows to define 1 element of the kernel matrix of a
    Hawkes process
    """
    _attrinfos = {"_kernel": {"writable": False}}

    def __init__(self):
        Base.__init__(self)
        self._kernel = _HawkesKernel()

    def is_zero(self):
        """Returns if this kernel is equal to 0
        """
        return self._kernel.is_zero()

    def get_support(self):
        """Returns the upperbound of the support
        """
        return self._kernel.get_support()

    def get_plot_support(self):
        """Returns support used to plot the kernel
        """
        return self._kernel.get_plot_support()

    def get_value(self, t):
        """Returns the value of the kernel at t
        """
        return self._kernel.get_value(t)

    def get_values(self, t_values):
        """Returns the value of the kernel for all times in t_values
        """
        return self._kernel.get_values(t_values)

    def get_norm(self, n_steps=10000):
        """Computes L1 norm

        Parameters
        ----------
        n_steps : `int`
            number of steps used for integral discretization

        Notes
        -----
        By default it approximates Riemann sum with step-wise function.
        It might be overloaded if L1 norm closed formula exists
        """
        return self._kernel.get_norm(n_steps)
