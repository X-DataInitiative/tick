# License: BSD 3 clause
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt


def toy_auc(x):
    if x < 1e-5: return toy_auc(1e-5) * 0.99
    if x > 1e2: return toy_auc(1e2) * 1.01
    return 0.5 - 0.1 * np.arctan(5 * np.log10(x / 1e-3))


def toy_kendall(x):
    if x == 0:
        return toy_kendall(1e-100)
    a = np.exp(-3 * np.abs(np.log10(x / 1e-3)))
    b = np.cos(np.log10(x / 1e-3))
    return a * b


def toy_kendall_no_diag(x):
    if x > 1e-1:
        return np.nan
    return toy_kendall(x * 1e3)


def get_toy_metrics():
    toy_metrics = OrderedDict()
    toy_metrics["alpha_auc"] = {'evaluator': toy_auc, 'best': 'max'}
    toy_metrics["estimation_error"] = {'evaluator': toy_auc, 'best': 'min'}
    toy_metrics["kendall"] = {'evaluator': toy_kendall, 'best': 'max'}
    toy_metrics["kendall_no_diag"] = {
        'evaluator': lambda x: toy_kendall(x * 1e-3),
        'best': 'max'
    }
    return toy_metrics


def get_toy_metrics_2d():

    def to2d(f1, f2=None):
        if f2 is None:
            f2 = f1
        return lambda x, y: f1(x) + f2(y)

    toy_metrics = OrderedDict()
    toy_metrics["alpha_auc"] = {
        'evaluator': to2d(toy_auc, toy_kendall), 'best': 'max'}
    toy_metrics["estimation_error"] = {
        'evaluator': to2d(toy_auc), 'best': 'min'}
    toy_metrics["kendall"] = {
        'evaluator': to2d(toy_kendall), 'best': 'max'}
    toy_metrics["kendall_no_diag"] = {
        'evaluator': to2d(toy_kendall, lambda x: toy_kendall(x * 1e-3)),
        'best': 'max'
    }

    return toy_metrics


if __name__ == '__main__':
    toy_auc = np.vectorize(toy_auc)
    xaxis = np.hstack((0, np.logspace(-9, 4)))
    yaxis = toy_auc(xaxis)
    ax = plt.subplot(211)
    ax.plot(xaxis, yaxis)
    ax.set_xscale('symlog', linthreshx=1e-10)
    ax.set_title('toy_auc')

    toy_kendall = np.vectorize(toy_kendall)

    xaxis = np.hstack((0, np.logspace(-9, 4)))
    yaxis = toy_kendall(xaxis)
    ax = plt.subplot(212)
    ax.plot(xaxis, yaxis)
    ax.set_xscale('symlog', linthreshx=1e-10)
    ax.set_title('toy_kendall')

    toy_estimation_error = toy_auc

    plt.show()
