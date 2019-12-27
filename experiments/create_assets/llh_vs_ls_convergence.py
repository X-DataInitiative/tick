import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tick.prox import ProxL1
from tick.solver import GD, AGD

import pickle

sys.path = [os.path.abspath('../../../')] + sys.path
sys.path = [os.path.abspath('../../')] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs, DECAYS_3
from tick.hawkes import (
    SimuHawkesSumExpKernels, ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik)


def run_simu_and_timings(adjacency, decays, baseline, end_time, prox,
                         history_file_path):
    simu = SimuHawkesSumExpKernels(
        adjacency, decays, baseline=baseline, end_time=end_time,
        verbose=False, seed=2111)

    simu.simulate()

    timestamps = simu.timestamps

    solver_kwargs = {
        'max_iter': 1300,
        'print_every': 100,
        'record_every': 20,
        'tol': 0,
    }

    model_ls = ModelHawkesSumExpKernLeastSq(DECAYS_3)
    model_ls.fit(timestamps)
    model_ls._model.compute_weights()

    model_llh = ModelHawkesSumExpKernLogLik(DECAYS_3)
    model_llh.fit(timestamps)
    model_llh._model.compute_weights()

    coeffs0 = 0.001 * np.ones(model_llh.n_coeffs)

    ista_llh = GD(linesearch=True, step=1e-4, **solver_kwargs)
    ista_llh.set_model(model_llh).set_prox(prox)
    ista_llh.solve(coeffs0)

    fista_llh = AGD(linesearch=True, step=1e-4, **solver_kwargs)
    fista_llh.set_model(model_llh).set_prox(prox)
    fista_llh.solve(coeffs0)

    ista_ls_kwargs = solver_kwargs.copy()
    ista_ls_kwargs['max_iter'] = 3000
    ista_ls = GD(linesearch=True, step=1e-4, **ista_ls_kwargs)
    ista_ls.set_model(model_ls).set_prox(prox)
    ista_ls.solve(coeffs0)

    fista_ls = AGD(linesearch=True, step=1e-4, **solver_kwargs)
    fista_ls.set_model(model_ls).set_prox(prox)
    fista_ls.solve(coeffs0)

    with open(history_file_path, 'wb') as output_file:
        histories = [solver.history
                     for solver in [ista_llh, fista_llh, ista_ls, fista_ls]]
        pickle.dump((simu.n_total_jumps, histories), output_file)


if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 3
    end_time_ = 5000
    prox_ = ProxL1(0, positive=True)

    original_coeffs_file_path_ = os.path.join(
        os.path.dirname(__file__),
        'original_coeffs_dim_{}_decays_{}.npy'.format(dim_, n_decays_))
    originalcoeffs_ = np.load(original_coeffs_file_path_, allow_pickle=True)
    baseline, adjacency = mus_alphas_from_coeffs(originalcoeffs_, n_decays_)

    history_file_path_ = 'llh_vs_ls_convergence_speed_T_{}.pkl'.format(end_time_)
    run_simu_and_timings(adjacency, DECAYS_3, baseline, end_time_, prox_,
                         history_file_path_)

    with open(history_file_path_, 'rb') as input_file:
        n_total_jumps, histories = pickle.load(input_file)

    labels = ['ISTA log likelihood', 'FISTA log likelihood',
              'ISTA least-squares', 'FISTA least-squares']

    colors_marker = [('C0', 'x'), ('C0', 'o'), ('C1', 'x'), ('C1', 'o')]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)

    ax_iter = axes[0]
    ax_time = axes[1]

    keep_indices_iter = 1000
    remove_indices_time = 300

    for label, history, (color, marker) in zip(labels, histories, colors_marker):
        objective_values = np.array(history.values['obj'])
        dist_obj = objective_values - objective_values.min()

        iterations = np.array(history.values['n_iter'])
        kept_index_iter = iterations <= keep_indices_iter

        markevery_iter = int(sum(kept_index_iter) / 8)
        ax_iter.plot(iterations[kept_index_iter], dist_obj[kept_index_iter],
                     color=color, marker=marker, markevery=markevery_iter)

        kept_index_time = iterations <= (iterations.max() - remove_indices_time)

        times = np.array(history.values['time'])

        markevery_time = int(sum(kept_index_time) / 8)
        ax_time.plot(times[kept_index_time], dist_obj[kept_index_time],
                     label=label, color=color, marker=marker,
                     markevery=markevery_time)

    ax_iter.set_yscale('log')
    ax_iter.set_xlabel('iterations', fontsize=14)
    ax_iter.set_ylabel('Distance to optimum', fontsize=14)

    ax_time.set_yscale('log')
    ax_time.set_xlabel('time', fontsize=14)
    ax_time.legend()

    fig.suptitle('$T = {}$ ({} events)\n'.format(end_time_, n_total_jumps),
                 fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.75])

    plt.show()
