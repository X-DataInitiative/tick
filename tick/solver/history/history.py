# License: BSD 3 clause

from collections import defaultdict
import dill

import numpy as np
from tick.base import Base
from numpy.linalg import norm


def spars_func(coeffs, **kwargs):
    eps = np.finfo(coeffs.dtype).eps
    return np.sum(np.abs(coeffs) > eps, axis=None)


class History(Base):
    """A class to manage the history along iterations of a solver

    Attributes
    ----------
    print_order : `list` or `str`
        The list of values to print along iterations

    values : `dict`
        A `dict` containing the history. Key is the value name and
        values are the values taken along the iterations

    last_values : `dict`
        A `dict` containing all the last history values

    _minimum_col_width : `int`
        Minimal size of a column when printing the history

    _minimizer : `None` or `numpy.ndarray`
        The minimizer of the objective. `None` if not specified.
        This is useful to compute a distance to the optimum.

    _minimum : `None` or `float`
        The minimal (optimal) value of the objective. `None` if not
        specified. This is useful to compute a distance to the optimum.

    _print_style : `list` or `str`
        The display style of all printed numbers

    _history_func : `dict`
        A dict given for all values the function to be applied before
        saving and displaying in history. This is useful for computing
        the sparsity, the rank, among other things, of the iterates
        along iterations of the solver

    _n_iter : `int`
        The current iteration number

    _col_widths : `list` or `int`
        A list containing the computed width of each column used for
        printing the history, based on the name length of the column
    """

    _attrinfos = {
        "values": {
            "writable": False
        },
        "last_values": {
            "writable": False
        },
    }

    def __init__(self):
        Base.__init__(self)
        self._minimum_col_width = 9
        self.print_order = ["n_iter", "obj", "step", "rel_obj"]
        # Instantiate values of the history
        self._clear()

        self._minimizer = None
        self._minimum = None
        self._set("values", None)
        self._col_widths = None
        self._n_iter = None

        # History function to compute history values based on parameters
        # used in a solver
        history_func = {}
        self._history_func = history_func

        # Default print style of history values. Default is %.2e
        print_style = defaultdict(lambda: "%.2e")
        print_style["n_iter"] = "%d"
        print_style["n_epoch"] = "%d"
        print_style["n_inner_prod"] = "%d"
        print_style["spars"] = "%d"
        print_style["rank"] = "%d"
        self._print_style = print_style

    def _clear(self):
        """Reset history values"""
        self._set("values", defaultdict(list))

    def _update(self, **kwargs):
        """Update the history along the iterations.

        For each keyword argument, we apply the history function corresponding
        to this keyword, and use its results in the history
        """
        self._n_iter = kwargs["n_iter"]
        history_func = self._history_func
        # We loop on both, history functions and kerword arguments
        keys = set(kwargs.keys()).union(set(history_func.keys()))
        for key in keys:
            # Either it has a corresponding history function which we
            # apply on all keywords
            if key in history_func:
                func = history_func[key]
                self.values[key].append(func(**kwargs))
            # Either we only record the value
            else:
                value = kwargs[key]
                self.values[key].append(value)

    def _format(self, name, index):
        try:
            formatted_str = self._print_style[name] % \
                            self.values[name][index]
        except TypeError:
            formatted_str = str(self.values[name][index])
        return formatted_str

    def _print_header(self):
        min_width = self._minimum_col_width
        line = ' | '.join(
            list([
                name.center(min_width) for name in self.print_order
                if name in self.values
            ]))
        names = [name.center(min_width) for name in self.print_order]
        self._col_widths = list(map(len, names))
        print(line)

    def _print_line(self, index):
        line = ' | '.join(
            list([
                self._format(name, index).rjust(self._col_widths[i])
                for i, name in enumerate(self.print_order)
                if name in self.values
            ]))
        print(line)

    def _print_history(self):
        """Verbose the current line of history
        """
        # If this is the first iteration, plot the history's column
        # names
        if self._col_widths is None:
            self._print_header()

        self._print_line(-1)

    def print_full_history(self):
        """Verbose the whole history
        """
        self._print_header()
        n_lines = len(next(iter(self.values.values())))

        for i in range(n_lines):
            self._print_line(i)

    @property
    def last_values(self):
        last_values = {}
        for key, hist in self.values.items():
            last_values[key] = hist[-1]
        return last_values

    def set_minimizer(self, minimizer: np.ndarray):
        """Set the minimizer of the objective, to compute distance
        to it along iterations

        Parameters
        ----------
        minimizer : `numpy.ndarray`, shape=(n_coeffs,)
            The minimizer of the objective

        Notes
        -----
        This adds dist_coeffs in history (distance to the minimizer)
        which is printed along iterations
        """
        self._minimizer = minimizer.copy()
        self._history_func["dist_coeffs"] = \
            lambda x, **kwargs: norm(x - self._minimizer)
        print_order = self.print_order
        if "dist_coeffs" not in print_order:
            print_order.append("dist_coeffs")

    def set_minimum(self, minimum: float):
        """Set the minimum of the objective, to compute distance to the
        optimum along iterations

        Parameters
        ----------
        minimum : `float`
            The minimizer of the objective

        Notes
        -----
        This adds dist_obj in history (distance to the minimum) which
        is printed along iterations
        """
        self._minimum = minimum
        self._history_func["dist_obj"] = \
            lambda obj, **kwargs: obj - self._minimum
        print_order = self.print_order
        if "dist_obj" not in print_order:
            print_order.append("dist_obj")

    def _as_dict(self):
        dd = Base._as_dict(self)
        dd.pop("values", None)
        return dd

    # We use dill for serialization because history uses lambda functions
    def __getstate__(self):
        return dill.dumps(self.__dict__)

    def __setstate__(self, state):
        object.__setattr__(self, '__dict__', dill.loads(state))
