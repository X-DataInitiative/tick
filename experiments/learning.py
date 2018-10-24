import datetime
import itertools
import pickle
import time
import traceback
from multiprocessing.pool import Pool

import numpy as np

from experiments.dict_utils import nested_update
from experiments.grid_search import get_new_range, plot_all_metrics, \
    get_best_point
from experiments.report_utils import get_image_directory, record_metrics

from experiments.metrics import get_metrics
from experiments.weights_computation import extract_index, logger, load_models
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
    i = int(extract_index(model_file_name, 'precomputed', 'pkl'))
    with open(model_file_name, 'rb') as model_file:
        model = pickle.load(model_file)

    # prepare information store
    info = {'train_loss': {}}

    for strength in strength_range:
        # Reinitialize problem (no warm start)
        coeffs = 1 * np.ones(model.n_coeffs)

        prox = create_prox(strength, model)
        solver = prepare_solver(SolverClass, solver_kwargs, model, prox)

        coeffs = solver.solve(coeffs)

        # warn if convergence was poor
        tol = solver_kwargs['tol']
        rel_objectives = solver.history.values['rel_obj']
        if len(rel_objectives) > 0:
            last_rel_obj = rel_objectives[-1]
            if last_rel_obj > (tol * 1.1):
                logger('did not converge train={} strength={:.3g}, '
                       'stopped at {:.3g} for tol={:.3g}'
                       .format(i, strength, last_rel_obj, tol))
        else:
            logger('failed for train={} strength={:.3g}, stopped at '
                   'first iteration'.format(i, strength))

        # record losses
        train_loss = model.loss(coeffs)
        info['train_loss'][strength] = train_loss

        compute_metrics(original_coeffs, coeffs, info, strength)

    return i, info


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


def learn_one_strength_range(dim, run_time, n_models, strength_range,
                             prox_info, solver_kwargs, n_cpu, directory_path):
    start = time.time()
    original_coeffs, model_file_paths = \
        load_models(dim, run_time, n_models, directory_path)

    SolverClass = GFB if prox_info['dim'] == 2 else AGD

    create_prox = prox_info['create_prox']
    infos = learn_in_parallel(strength_range, create_prox, compute_metrics,
                              original_coeffs, SolverClass, solver_kwargs,
                              model_file_paths, n_cpu)
    logger('needed', time.time() - start, 'seconds')

    return infos


def find_best_metrics_1d(dim, run_time, n_models, prox_info, solver_kwargs,
                         n_cpu, directory_path, max_run_count=10):
    prox_name = prox_info['name']
    logger('### For prox %s' % prox_name)

    run_strength_range = np.hstack(
        (0, np.logspace(-7, 2, prox_info['n_initial_points'])))

    aggregated_run_infos = {}
    for run_count in range(max_run_count):
        logger('{} Run {} - With {} new points'
               .format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M'),
                       run_count, len(run_strength_range)))

        run_infos = learn_one_strength_range(
            dim, run_time, n_models, run_strength_range,
            prox_info, solver_kwargs, n_cpu, directory_path)

        aggregated_run_infos = nested_update(run_infos, aggregated_run_infos)

        run_strength_range = get_new_range(
            aggregated_run_infos, get_metrics(), prox_info)

        if len(run_strength_range) == 0:
            break

        if run_count == max_run_count:
            logger('Maximum number of run reached, run_strength_range was {}'
                   .format(run_strength_range))

    # record_metrics(aggregated_run_infos, dim, run_time, n_models, prox_name,
    #                logger, suffix)

    return aggregated_run_infos


if __name__ == '__main__':
    from pprint import pprint
    from experiments.tested_prox import create_prox_l1_no_mu
    from experiments.metrics import compute_metrics, get_metrics
    from experiments.grid_search import get_new_range, get_best_point

    dim = 30
    run_time = 500
    n_models = 3
    directory_path = '/Users/martin/Downloads/jmlr_hawkes_data/'

    solver_kwargs = {'tol': 1e-4, 'max_iter': 1000}
    original_coeffs, model_file_paths = \
        load_models(dim, run_time, n_models, directory_path)

    strength_range = [1e-3, 1e-2]

    create_prox = create_prox_l1_no_mu

    _, info = learn_one_model(
        model_file_paths[0], strength_range, create_prox, compute_metrics,
        original_coeffs, AGD, solver_kwargs)

    print('\n---- one model ----')
    pprint(info)

    prox_info = {
        'name': 'prox_l1',
        'n_initial_points': 10,
        'max_relative_step': 1.4,
        'create_prox': create_prox_l1_no_mu,
        'tol': 1e-10,
        'dim': 1,
    }

    n_cpu = 3
    infos = learn_in_parallel(
        strength_range, create_prox, compute_metrics, original_coeffs,
        AGD, solver_kwargs, model_file_paths, n_cpu)

    print('\n---- in parallel ----')
    pprint(infos)

    infos = find_best_metrics_1d(
        dim, run_time, n_models, prox_info, solver_kwargs,
        n_cpu, directory_path, max_run_count=10)

    print('\n---- find best metrics ----')
    metrics = get_metrics()
    for metric in metrics:
        best_point = get_best_point(metrics, metric, infos)
        print(metric, best_point)
