import datetime
import itertools
import pickle
import time
import traceback
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np

from experiments.dict_utils import nested_update

from experiments.grid_search_2d import get_new_range_2d
from experiments.report_utils import record_metrics, logger
from experiments.metrics import get_metrics, compute_metrics
from experiments.grid_search_1d import get_new_range
from experiments.tested_prox import get_n_decays_from_model

from experiments.weights_computation import extract_index, load_models
from tick.solver import AGD, GFB


def prepare_solver(SolverClass, solver_kwargs, model, prox):
    if SolverClass == AGD:
        solver_kwargs['linesearch'] = False
    solver_kwargs['step'] = 1. / model.get_lip_best()
    solver_kwargs['verbose'] = False
    solver = SolverClass(**solver_kwargs)
    solver.set_model(model)
    solver.set_prox(prox)
    return solver


def learn_one_model(model_file_name, strength_range, create_prox,
                    compute_metrics, original_coeffs, SolverClass,
                    solver_kwargs):
    model_index = int(extract_index(model_file_name, 'precomputed', 'pkl'))
    with open(model_file_name, 'rb') as model_file:
        model = pickle.load(model_file)

    # prepare information store
    info = {'train_loss': {}}

    for strength in strength_range:
        # Reinitialize problem (no warm start)
        coeffs = 1 * np.ones(model.n_coeffs)

        prox = create_prox(strength, model, logger)
        solver = prepare_solver(SolverClass, solver_kwargs, model, prox)

        coeffs = solver.solve(coeffs)

        # warn if convergence was poor
        tol = solver_kwargs['tol']
        rel_objectives = solver.history.values['rel_obj']
        if len(rel_objectives) > 0:
            last_rel_obj = rel_objectives[-1]
            last_time = solver.history.values['time'][-1]
            if last_rel_obj > (tol * 1.1):
                if isinstance(strength, tuple):
                    strength_str = ', '.join(
                        '{:.3g}'.format(s) for s in strength)
                else:
                    strength_str = '{:.3g}'.format(strength)
                logger('{} - did not converge train={} strength={}, '
                       'stopped at {:.3g} for tol={:.3g}, took {:.3g}s'
                       .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M'), 
                               model_index, strength_str, last_rel_obj, tol, last_time))
        else:
            logger('failed for train={} strength={:.3g}, stopped at '
                   'first iteration'.format(model_index, strength))

        # record losses
        train_loss = model.loss(coeffs)
        info['train_loss'][strength] = train_loss

        compute_metrics(original_coeffs, coeffs,
                        get_n_decays_from_model(model), info, strength)

    return model_index, info


def learn_in_parallel(strength_range, create_prox, compute_metrics,
                      original_coeffs, SolverClass, solver_kwargs,
                      model_file_names, n_cpu):
    args_list = model_file_names
    static_args = [strength_range, create_prox, compute_metrics,
                   original_coeffs, SolverClass, solver_kwargs]
    args_list = [[arg] + static_args for arg in args_list]

    n_cpu = min(n_cpu, len(model_file_names))
    pool = Pool(n_cpu)
    infos = None
    try:
        info_list = pool.starmap(learn_one_model, args_list)
        infos = {i: info for (i, info) in info_list}
        pool.close()
        pool.join()
    except Exception:
        traceback.print_exc()
        logger(traceback.format_exc())
    finally:
        pool.terminate()
    return infos


def learn_one_strength_range(original_coeffs, model_file_paths, strength_range,
                             prox_info, solver_kwargs, n_cpu):
    start = time.time()

    SolverClass = GFB if prox_info['dim'] == 2 else AGD

    create_prox = prox_info['create_prox']
    infos = learn_in_parallel(strength_range, create_prox, compute_metrics,
                              original_coeffs, SolverClass, solver_kwargs,
                              model_file_paths, n_cpu)
    logger('needed {:.2f} seconds'.format(time.time() - start))

    return infos


def find_best_metrics_1d(original_coeffs, model_file_paths,
                         prox_info, solver_kwargs, n_cpu, max_run_count=10):
    run_strength_range = np.hstack(
        (0, np.logspace(-7, 2, prox_info['n_initial_points'])))

    aggregated_run_infos = {}
    for run_count in range(max_run_count):
        logger('{} Run {} - With {} new points'
               .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M'),
                       run_count, len(run_strength_range)))

        run_infos = learn_one_strength_range(
            original_coeffs, model_file_paths, run_strength_range,
            prox_info, solver_kwargs, n_cpu)

        aggregated_run_infos = nested_update(run_infos, aggregated_run_infos)

        run_strength_range = get_new_range(
            aggregated_run_infos, get_metrics(), prox_info)

        if len(run_strength_range) == 0:
            break

        if run_count == max_run_count:
            logger('Maximum number of run reached, run_strength_range was {}'
                   .format(run_strength_range))

    return aggregated_run_infos


def find_best_metrics_2d(original_coeffs, model_file_paths,
                         prox_info, solver_kwargs, n_cpu, max_run_count=10):
    n_initial_points = prox_info['n_initial_points']
    toy_strength_range_1 = np.logspace(-8, -2, n_initial_points)
    toy_strength_range_2 = np.logspace(-8, -2, n_initial_points)
    run_strength_range = [(l1, tau) for (l1, tau)
                          in itertools.product(toy_strength_range_1,
                                               toy_strength_range_2)]

    aggregated_run_infos = {}
    for run_count in range(max_run_count):
        logger('{} Run {} - With {} new points'
               .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M'),
                       run_count, len(run_strength_range)))

        run_infos = learn_one_strength_range(
            original_coeffs, model_file_paths, run_strength_range,
            prox_info, solver_kwargs, n_cpu)

        aggregated_run_infos = nested_update(run_infos, aggregated_run_infos)

        run_strength_range = get_new_range_2d(
            aggregated_run_infos, prox_info, get_metrics(), extension_step=1e2)

        if len(run_strength_range) == 0:
            break

        if run_count == max_run_count:
            logger('Maximum number of run reached, run_strength_range was {}'
                   .format(run_strength_range))

    return aggregated_run_infos


def find_best_metrics(dim, run_time, n_decays, n_models, prox_info,
                      solver_kwargs, directory_path, n_cpu=-1, max_run_count=10,
                      suffix=''):
    if n_cpu < 1:
        n_cpu = cpu_count()
    
    if 'tol' in prox_info:
        solver_kwargs['tol'] = prox_info['tol']

    prox_name = prox_info['name']
    logger('### For prox %s' % prox_name)

    original_coeffs, model_file_paths = \
        load_models(dim, run_time, n_decays, n_models, directory_path)

    func = find_best_metrics_1d if prox_info['dim'] == 1 \
        else find_best_metrics_2d

    infos = func(original_coeffs, model_file_paths,
                 prox_info, solver_kwargs, n_cpu, max_run_count=max_run_count)

    record_metrics(infos, dim, run_time, n_models, prox_name, prox_info['dim'],
                   solver_kwargs['tol'], logger, suffix)


if __name__ == '__main__':
    from pprint import pprint
    from experiments.tested_prox import create_prox_l1w_no_mu_nuclear
    from experiments.grid_search_1d import get_best_point

    dim_ = 30
    n_decays_ = 3
    run_time_ = 500
    n_models_ = 3
    directory_path_ = '/Users/martin/Downloads/jmlr_hawkes_data/'

    solver_kwargs_ = {'tol': 1e-4, 'max_iter': 1000}
    original_coeffs_, model_file_paths_ = \
        load_models(dim_, run_time_, n_decays_, n_models_, directory_path_)

    strength_range_ = [(1e-3, 1e-1), (1e-2, 1e-2)]

    create_prox_ = create_prox_l1w_no_mu_nuclear

    _, info_ = learn_one_model(
        model_file_paths_[0], strength_range_, create_prox_, compute_metrics,
        original_coeffs_, GFB, solver_kwargs_)

    print('\n---- one model ----')
    pprint(info_)
    #
    prox_info_ = {
        'name': 'l1_w_nuclear',
        'n_initial_points': 3,
        'max_relative_step': 1.4,
        'create_prox': create_prox_l1w_no_mu_nuclear,
        'tol': 1e-4,
        'dim': 2,
    }

    # n_cpu = 3
    # infos = learn_in_parallel(
    #     strength_range, create_prox, compute_metrics, original_coeffs,
    #     AGD, solver_kwargs, model_file_paths, n_cpu)
    #
    # print('\n---- in parallel ----')
    # pprint(infos)
    #
    # infos = find_best_metrics_1d(
    #     dim, run_time, n_models, prox_info, solver_kwargs,
    #     n_cpu, directory_path, max_run_count=10)
    #
    # print('\n---- find best metrics ----')
    # metrics = get_metrics()
    # for metric in metrics:
    #     best_point = get_best_point(metrics, metric, infos)
    #     print(metric, best_point)

    find_best_metrics(dim_, run_time_, n_decays_, n_models_, prox_info_,
                      solver_kwargs_, directory_path_, n_cpu=1,
                      max_run_count=10, suffix='test')
