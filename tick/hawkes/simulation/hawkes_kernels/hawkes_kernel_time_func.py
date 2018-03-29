# License: BSD 3 clause

from tick.hawkes.simulation.build.hawkes_simulation import (
    HawkesKernelTimeFunc as _HawkesKernelTimeFunc)
from .hawkes_kernel import HawkesKernel


class HawkesKernelTimeFunc(HawkesKernel):
    """Hawkes kernel defined by an arbitrary time function.
    
    Parameters
    ----------
    time_function : `tick.base.TimeFunc`
        Time function defining the kernel.
        
    t_values : `np.ndarray` shape=(n_points,)
        time array  used to build the time function. Might be given together 
        with `y_values` instead of `time_function`.
        
    y_values : `np.ndarray` shape=(n_points,)
        values array used to build the time function. Might be given together 
        with `t_values` instead of `time_function`.
    """

    def __init__(self, time_function=None, t_values=None, y_values=None):
        HawkesKernel.__init__(self)
        if (t_values is None and time_function is None) \
                or (t_values is not None and time_function is not None):
            raise ValueError("Either time_function or "
                             "t_values / y_values must be filled")

        if t_values is not None and y_values is None:
            raise ValueError("t_values and y_values must be filled")

        if time_function is not None:
            self._kernel = _HawkesKernelTimeFunc(time_function._time_function)
        else:
            self._kernel = _HawkesKernelTimeFunc(t_values, y_values)

    @property
    def time_function(self):
        return self._kernel.get_time_function()

    def __str__(self):
        return "KernelTimeFunc"

    def __repr__(self):
        return "KernelTimeFunc"

    def __strtex__(self):
        return "TimeFunc Kernel"
