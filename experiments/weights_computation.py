# License: BSD 3 clause

import datetime
import os
import pickle
import re
import traceback
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np

from experiments.hawkes_coeffs import retrieve_coeffs, coeffs_from_mus_alpha, \
    mus_alphas_from_coeffs, DECAY_1, DECAYS_3
from experiments.io_utils import get_precomputed_models_dir, load_directory, \
    get_simulation_dir
from experiments.report_utils import logger
from experiments.simulation import get_simulation_files
from tick.hawkes import ModelHawkesExpKernLeastSq, ModelHawkesSumExpKernLeastSq, ModelHawkesExpKernLogLik, \
    ModelHawkesSumExpKernLogLik

PRECOMPUTED_PREFIX = 'precomputed'

LEAST_SQ_LOSS = 'LEAST_SQ_LOSS'
LLH_LOSS = 'LLH_LOSS'

report_for_pre_compute_file_name = None


def get_precomputed_models_files(dim, run_time, n_decays, directory_prefix, loss):
    directory = get_precomputed_models_dir(dim, run_time, n_decays,
                                           directory_prefix, loss)
    return load_directory(directory, 'pkl')


def extract_index(file_name, pattern, extension):
    regex = '%s_(\d+).%s' % (pattern, extension)
    match = re.search(regex, file_name)
    return match.group(1)


def precompute_weights(dim, run_time, n_decays, simulation_file,
                       directory_prefix, loss):
    simulation_path = os.path.join(
        get_simulation_dir(dim, run_time, n_decays, directory_prefix),
        simulation_file)

    # we must convert it to a list of numpy arrays
    ticks = list(np.load(simulation_path, allow_pickle=True))

    # fit ticks and precompute weights
    if n_decays == 1:
        if loss == LEAST_SQ_LOSS:
            model = ModelHawkesExpKernLeastSq(DECAY_1, n_threads=1).fit(ticks)
        else:
            model = ModelHawkesExpKernLogLik(DECAY_1, n_threads=1).fit(ticks)
    else:
        if loss == LEAST_SQ_LOSS:
            model = ModelHawkesSumExpKernLeastSq(DECAYS_3, n_threads=1).fit(ticks)
        else:
            model = ModelHawkesSumExpKernLogLik(DECAYS_3, n_threads=1).fit(ticks)
            model._least_sq_model = ModelHawkesSumExpKernLeastSq(DECAYS_3, n_threads=1).fit(ticks)
            model._least_sq_model.loss(np.ones(model.n_coeffs))

    model_loss = model.loss(np.ones(model.n_coeffs))
    lip_best = model.get_lip_best() if loss == LEAST_SQ_LOSS else -1
    index = extract_index(simulation_file, 'simulation', 'npy')

    precomputed_models_dir = \
        get_precomputed_models_dir(dim, run_time, n_decays, directory_prefix, loss)
    os.makedirs(precomputed_models_dir, exist_ok=True)

    precompute_file_name = '{}_{}.pkl'.format(PRECOMPUTED_PREFIX, index)
    precomputed_model_path = os.path.join(
        precomputed_models_dir, precompute_file_name)

    with open(precomputed_model_path, 'wb') as save_file:
        pickle.dump(model, save_file)

    logger('saved {} , with loss = {}, lipbest={}'.format(
        precompute_file_name, model_loss, lip_best))


def pre_compute_hawkes(dim, run_time, n_decays,
                       max_pre_computed_hawkes, directory_prefix,  n_cpu=-1, loss=LEAST_SQ_LOSS):
    if n_cpu < 1:
        n_cpu = cpu_count()

    available_simulations = get_simulation_files(dim, run_time, n_decays,
                                                 directory_prefix)
    if len(available_simulations) < max_pre_computed_hawkes:
        logger('WARNING FOR dim={}, T={:.0f}, only {} simulations available'
               .format(dim, run_time, len(available_simulations)))

    already_precomputed_index = [
        extract_index(precomputed_file, 'precomputed', 'pkl')
        for precomputed_file in
        get_precomputed_models_files(dim, run_time, n_decays, directory_prefix, loss)
    ]

    to_compute_simulations = [
        simulation_file
        for simulation_file in available_simulations
        if extract_index(simulation_file, 'simulation', 'npy')
           not in already_precomputed_index
    ]

    selected_simulations = [
        simulation_file
        for simulation_file in to_compute_simulations
        if int(extract_index(simulation_file, 'simulation', 'npy'))
           < max_pre_computed_hawkes
    ]

    now = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    logger('\n' + '-' * 30)
    logger('DIM={} T={:.0f} : {}'.format(dim, run_time, now))
    if len(selected_simulations) > 0:
        args = [(dim, run_time, n_decays, simulation_file, directory_prefix, loss)
                for simulation_file in selected_simulations]
        pool = Pool(min(n_cpu, len(selected_simulations)))
        try:
            pool.starmap(precompute_weights, args)
            pool.close()
            pool.join()
        except Exception:
            logger(traceback.format_exc())
            traceback.print_exc()
        finally:
            pool.terminate()
    else:
        logger('{} models already computed'
              .format(len(already_precomputed_index)))


def load_models(dim, run_time, n_decays, n_models, directory_prefix, loss):
    precomputed_models_dir = \
        get_precomputed_models_dir(dim, run_time, n_decays, directory_prefix, loss)

    _, mu, alpha = retrieve_coeffs(dim, n_decays, directory_prefix)
    original_coeffs = coeffs_from_mus_alpha(mu, alpha)

    model_file_names = load_directory(precomputed_models_dir, 'pkl')
    logger('Retrieved {} precomputed models'.format(len(model_file_names)))

    if len(model_file_names) > n_models:
        model_file_names = model_file_names[:n_models]
        logger('We keep {} precomputed models'.format(len(model_file_names)))

    if len(model_file_names) < n_models:
        logger('Only {} precomputed models'.format(len(model_file_names)))

    model_file_paths = [
        os.path.join(precomputed_models_dir, model_file_name)
        for model_file_name in model_file_names]
    return original_coeffs, model_file_paths


if __name__ == '__main__':
    from experiments.plot_hawkes import plot_coeffs

    print('extract_index',
          extract_index('simulation_003.npy', 'simulation', 'npy'))

    dim_ = 30
    run_times_ = [500, 1000]
    max_pre_computed_hawkes_ = 10
    n_decays_ = 3

    directory_path_ = '/Users/martin/Downloads/jmlr_hawkes_data/'

    for run_time_ in run_times_:
        pre_compute_hawkes(dim_, run_time_, n_decays_,
                           max_pre_computed_hawkes_, directory_path_)

    original_coeffs_, model_file_paths_ = load_models(
        dim_, run_times_[0], n_decays_, 5, directory_path_, LEAST_SQ_LOSS)

    mu_, alpha_ = mus_alphas_from_coeffs(original_coeffs_, n_decays_)
    plot_coeffs(mu_, alpha_)
    print(model_file_paths_)

    for model_file_path_ in model_file_paths_:
        with open(model_file_path_, 'rb') as model_file:
            model = pickle.load(model_file)
            print(extract_index(model_file_path_, 'precomputed', 'pkl'),
                  'model.get_lip_best()', model.get_lip_best())
