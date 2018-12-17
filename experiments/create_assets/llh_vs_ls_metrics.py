import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from experiments.dict_utils import nested_update
from experiments.grid_search_1d import get_new_range, plot_all_metrics, \
    get_best_point
from experiments.learning import learn_one_model
from experiments.metrics_utils import extract_metric, mean_and_std, \
    strength_range_from_infos
from experiments.tested_prox import create_prox_l1_no_mu, \
    create_prox_l1w_no_mu, create_prox_l1w_no_mu_un
from tick.prox import ProxL1
from tick.solver import GD, AGD

import pickle

sys.path = [os.path.abspath('../../../')] + sys.path
sys.path = [os.path.abspath('../../')] + sys.path

from experiments.hawkes_coeffs import mus_alphas_from_coeffs, DECAYS_3, DECAY_1
from experiments.metrics import get_metrics, compute_metrics

from tick.hawkes import (
    SimuHawkesSumExpKernels, ModelHawkesSumExpKernLeastSq,
    ModelHawkesSumExpKernLogLik)


computed_metrics_names = ['estimation_error_no_diag', 'alphas_auc_no_diag',
                           'kendall_no_diag']

def create_models(adjacency, decays, baseline, end_time):
    simu = SimuHawkesSumExpKernels(
        adjacency, decays, baseline=baseline, end_time=end_time,
        verbose=False, seed=2111)

    simu.simulate()

    timestamps = simu.timestamps

    model_ls = ModelHawkesSumExpKernLeastSq(decays)
    model_ls.fit(timestamps)
    model_ls._model.compute_weights()

    model_llh = ModelHawkesSumExpKernLogLik(decays)
    model_llh.fit(timestamps)
    model_llh._model.compute_weights()

    return model_ls, model_llh


def find_best_metrics_1d(model, create_prox, original_coeffs, solver_kwargs):
    run_strength_range = np.hstack(
        (0, np.logspace(-4, -1, 5)))

    max_run_count = 10
    aggregated_run_infos = {}
    for run_count in range(max_run_count):
        print(('{} Run {} - With {} new points'
               .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M'),
                       run_count, len(run_strength_range))))

        run_infos = {
            0: learn_one_model(model, run_strength_range, create_prox,
                               compute_metrics, original_coeffs, AGD,
                               solver_kwargs, save_coeffs=False)[1]}

        aggregated_run_infos = nested_update(run_infos, aggregated_run_infos)

        run_strength_range = get_new_range(
            aggregated_run_infos, get_metrics(), {'max_relative_step': 1.3})

        if len(run_strength_range) == 0:
            break

        if run_count == max_run_count:
            print('Maximum number of run reached, run_strength_range was {}'
                  .format(run_strength_range))

    plot_all_metrics(aggregated_run_infos, get_metrics())
    plt.show()

    infos = aggregated_run_infos

    res = {}
    for metric in computed_metrics_names:
        strength_range = strength_range_from_infos(infos)
        metric_metrics = extract_metric(metric, infos)
        mean_metrics, std = mean_and_std(metric_metrics)
        best_point = get_best_point(get_metrics(), metric, infos)

        res[metric] = (strength_range[best_point], mean_metrics[best_point])

    return res


def get_full_history_file_path(model, metric, history_file_path):
    if isinstance(model, ModelHawkesSumExpKernLeastSq) or model == 'ls':
        model_name = 'model_ls'
    else:
        model_name = 'model_llh'

    return history_file_path.replace(
        '.pkl', '_{}_{}.pkl'.format(model_name, metric))


def run_simu_and_timings(model, metric, create_prox,
                         history_file_path, infos, solver_kwargs):
    solver_kwargs = solver_kwargs.copy()

    strength = infos[metric][0]
    prox = create_prox(strength, model)

    if 'step' in solver_kwargs:
        step = solver_kwargs['step']
        del solver_kwargs['step']
    else:
        step = 1. / model.get_lip_best()

    coeffs0 = 0.01 * np.ones(model.n_coeffs)

    ista = GD(linesearch=False, step=step, **solver_kwargs)
    ista.set_model(model).set_prox(prox)
    ista.solve(coeffs0)

    fista = AGD(linesearch=False, step=step, **solver_kwargs)
    fista.set_model(model).set_prox(prox)
    fista.solve(coeffs0)

    with open(get_full_history_file_path(model, metric, history_file_path),
              'wb') as output_file:
        histories = [solver.history
                     for solver in [ista, fista]]
        pickle.dump((model.n_jumps, histories), output_file)


def perform_cross_fold(model, solver_kwargs, create_prox, originalcoeffs):
    infos = find_best_metrics_1d(model, create_prox, originalcoeffs,
                                 solver_kwargs)

    if isinstance(model, ModelHawkesSumExpKernLeastSq):
        info_file_path = LEAST_SQUARES_INFO_PATH.format(end_time_)
    else:
        info_file_path = LOG_LIKELIHOOD_INFO_PATH.format(end_time_)

    with open(info_file_path, 'wb') as output_file:
        pickle.dump(infos, output_file)


def record_histories(model, create_prox, history_file_path, solver_kwargs):

    if isinstance(model, ModelHawkesSumExpKernLeastSq):
        info_file_path = LEAST_SQUARES_INFO_PATH.format(end_time_)
    else:
        info_file_path = LOG_LIKELIHOOD_INFO_PATH.format(end_time_)

    with open(info_file_path, 'rb') as input_file:
        infos = pickle.load(input_file)

    print('INFO', model.__class__, infos)

    for metric in computed_metrics_names:
        run_simu_and_timings(model, metric, create_prox,
                             history_file_path, infos, solver_kwargs)


LEAST_SQUARES_INFO_PATH = 'histories_1/least_squares_info_{}.pkl'
LOG_LIKELIHOOD_INFO_PATH = 'histories_1/log_likelihood_info_{}.pkl'

if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 1
    end_time_ = 20000

    original_coeffs_file_path_ = os.path.join(
        os.path.dirname(__file__),
        'original_coeffs_dim_{}_decays_{}.npy'.format(dim_, n_decays_))
    originalcoeffs_ = np.load(original_coeffs_file_path_)
    baseline, adjacency = mus_alphas_from_coeffs(originalcoeffs_, n_decays_)
    adjacency = adjacency.reshape(dim_, dim_, n_decays_)

    history_file_path_ = 'histories_1/T_{}.pkl'.format(end_time_)

    run_any = False
    if run_any:
        model_ls, model_llh = create_models(
            adjacency, np.array([DECAY_1]), baseline, end_time_)
        create_prox_ = create_prox_l1_no_mu
        solver_kwargs_ = {'max_iter': 10000, 'tol': 1e-10, 'record_every': 20,
                          'print_every': 100}
        solver_kwargs_llh_ = dict(step=1e-1, verbose=True, **solver_kwargs_)
        #
        # perform_cross_fold(model_ls, solver_kwargs_, create_prox_,
        #                    originalcoeffs_)
        # perform_cross_fold(model_llh, solver_kwargs_llh_, create_prox_,
        #                    originalcoeffs_)
        #
        # record_histories(model_ls, create_prox_, history_file_path_,
        #                  solver_kwargs_)
        # record_histories(model_llh, create_prox_, history_file_path_,
        #                  solver_kwargs_llh_)

    metric_histories = {}
    for metric in computed_metrics_names:
        llh_full_history_file_path_ = get_full_history_file_path(
            'llh', metric, history_file_path_)
        with open(llh_full_history_file_path_, 'rb') as input_file:
            n_total_jumps, histories = pickle.load(input_file)
            metric_histories[metric] = histories

        ls_full_history_file_path_ = get_full_history_file_path(
            'ls', metric, history_file_path_)
        with open(ls_full_history_file_path_, 'rb') as input_file:
            n_total_jumps, histories = pickle.load(input_file)
            metric_histories[metric] += histories

    labels = ['ISTA log likelihood', 'FISTA log likelihood',
              'ISTA least-squares', 'FISTA least-squares']

    colors_marker = [('C0', 'x'), ('C0', 'o'), ('C1', 'x'), ('C1', 'o')]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    ax_error = axes[0]
    ax_auc = axes[1]
    ax_kendall = axes[2]

    metrics = get_metrics(originalcoeffs_, n_decays_)

    for i, metric in enumerate(computed_metrics_names):
        ax = axes[i]
        histories = metric_histories[metric]

        for label, history, (color, marker) in zip(labels, histories,
                                                   colors_marker):

            iterations = np.array(history.values['n_iter'])
            iterates = np.array(history.values['x'])
            times = np.array(history.values['time'])

            metric_values = []
            for iterate in iterates:
                metric_values += [metrics[metric]['evaluator'](iterate)]

            markevery_time = int(len(iterations) / 8)

            ax.plot(times, metric_values,
                    label=label, color=color, marker=marker,
                    markevery=markevery_time)

            ax.set_xlabel('time (s)', fontsize=14)
            ax.set_xscale('log')

    ax_error.set_title('Estimation error', fontsize=14)
    ax_auc.set_title('AUC', fontsize=14)
    ax_kendall.set_title('Kendall', fontsize=14)
    legend = ax_kendall.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    # fig.suptitle('$T = {}$ ({} events)\n'.format(end_time_, n_total_jumps),
    #              fontsize=16)

    # plt.show()
    fig.tight_layout()
    fig.savefig('llh_vs_ls_metrics_200001.pdf', bbox_inches='tight',
                bbox_extra_artists=(legend, ))
