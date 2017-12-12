# License: BSD 3 clause

import unittest

import numpy as np

from tick.linear_model import LogisticRegression
from tick.plot import plot_history
from tick.solver import GD, AGD, History


class Test(unittest.TestCase):
    def setUp(self):
        np.random.seed(238924)

        self.n_iter1 = list(range(0, 30, 3))
        self.obj1 = [np.random.normal() for _ in range(len(self.n_iter1))]

        self.n_iter2 = list(range(2, 40, 2))
        self.obj2 = [np.random.normal() for _ in range(len(self.n_iter2))]

        self.solver1 = GD()
        history1 = History()
        history1._set("values", {'n_iter': self.n_iter1, 'obj': self.obj1})
        self.solver1._set("history", history1)

        self.solver2 = AGD()
        history2 = History()
        history2._set("values", {'n_iter': self.n_iter2, 'obj': self.obj2})
        self.solver2._set("history", history2)

    def test_plot_history_solver(self):
        """...Test plot_history rendering given a list of solvers
        """
        labels = ['solver 1', 'solver 2']
        fig = plot_history([self.solver1, self.solver2], show=False,
                           labels=labels)
        ax = fig.axes[0]

        ax_n_iter1, ax_obj1 = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter1, self.n_iter1)
        np.testing.assert_array_equal(ax_obj1, self.obj1)
        self.assertEqual(ax.lines[0].get_label(), labels[0])

        ax_n_iter2, ax_obj2 = ax.lines[1].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter2, self.n_iter2)
        np.testing.assert_array_equal(ax_obj2, self.obj2)
        self.assertEqual(ax.lines[1].get_label(), labels[1])

    def test_plot_history_solver_dist_min(self):
        """...Test plot_history rendering with dist_min argument
        """

        fig = plot_history([self.solver1, self.solver2], show=False,
                           dist_min=True)
        ax = fig.axes[0]

        min_obj = min(min(self.obj1), min(self.obj2))

        ax_n_iter1, ax_obj1 = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter1, self.n_iter1)
        np.testing.assert_array_equal(ax_obj1, np.array(self.obj1) - min_obj)
        self.assertEqual(ax.lines[0].get_label(), 'GD')

        ax_n_iter2, ax_obj2 = ax.lines[1].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter2, self.n_iter2)
        np.testing.assert_array_equal(ax_obj2, np.array(self.obj2) - min_obj)
        self.assertEqual(ax.lines[1].get_label(), 'AGD')

    def test_plot_history_solver_log_scale(self):
        """...Test plot_history rendering on a log scale
        """

        fig = plot_history([self.solver1, self.solver2], show=False,
                           dist_min=True, log_scale=True)
        ax = fig.axes[0]
        self.assertEqual(ax.yaxis.get_scale(), 'log')

    def test_plot_history_learner(self):
        """...Test plot_history rendering given a list of learners
        """
        learner1 = LogisticRegression(solver='svrg')
        learner1._solver_obj._set('history', self.solver1.history)
        learner2 = LogisticRegression(solver='agd')
        learner2._solver_obj._set('history', self.solver2.history)

        fig = plot_history([learner1, learner2], show=False)
        ax = fig.axes[0]

        ax_n_iter1, ax_obj1 = ax.lines[0].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter1, self.n_iter1)
        np.testing.assert_array_equal(ax_obj1, self.obj1)
        self.assertEqual(ax.lines[0].get_label(), 'SVRG')

        ax_n_iter2, ax_obj2 = ax.lines[1].get_xydata().T
        np.testing.assert_array_equal(ax_n_iter2, self.n_iter2)
        np.testing.assert_array_equal(ax_obj2, self.obj2)
        self.assertEqual(ax.lines[1].get_label(), 'AGD')


if __name__ == '__main__':
    unittest.main()
