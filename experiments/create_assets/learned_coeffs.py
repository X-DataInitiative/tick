import os
import sys

from experiments.create_assets.coeffs import plot_coeffs_3_decays
from experiments.hawkes_coeffs import mus_alphas_from_coeffs

sys.path = [os.path.abspath('../../')] + sys.path

from collections import OrderedDict

import pandas as pd
import numpy as np

from experiments.learning import learn_one_model
from experiments.metrics import compute_metrics
from experiments.tested_prox import create_prox_l1_no_mu, \
    create_prox_l1w_no_mu_un, create_prox_l1_no_mu_nuclear, \
    create_prox_l1w_un_no_mu_nuclear

from tick.solver import AGD, GFB

prox_infos = OrderedDict()

prox_infos['l1'] = {
    'create_prox': create_prox_l1_no_mu,
    'tol': 1e-10,
}

prox_infos['l1w_un'] = {
    'create_prox': create_prox_l1w_no_mu_un,
    'tol': 1e-10,
}


prox_infos['l1_nuclear'] = {
    'create_prox': create_prox_l1_no_mu_nuclear,
    'tol': 1e-8,
}

prox_infos['l1w_un_nuclear'] = {
    'create_prox': create_prox_l1w_un_no_mu_nuclear,
    'tol': 1e-8,
}


def learn_coeffs(dim, n_decays, end_time, prox_name):
    original_coeffs_file_path = os.path.join(
        os.path.dirname(__file__),
        'original_coeffs_dim_{}_decays_{}.npy'.format(dim, n_decays))

    model_file_path = os.path.join(
        os.path.dirname(__file__),
        'models/dim={}_u={}_T={}.pkl'.format(dim, n_decays, end_time))

    df = pd.read_csv('used_lambdas_n_decays_{}.csv'.format(n_decays))

    best_strengths = df[(df['end_time'] == end_time)
                        & (df['prox'] == prox_name)]

    if 'nuclear' in prox_name:
        best_strength = eval(best_strengths['estimation_error'].values[0])
    else:
        best_strength = float(best_strengths['estimation_error'])

    strength_range = [best_strength]

    create_prox = prox_infos[prox_name]['create_prox']

    original_coeffs = np.load(original_coeffs_file_path)

    if 'nuclear' in prox_name:
        SolverClass = GFB
    else:
        SolverClass = AGD

    solver_kwargs = {
        'max_iter': 9000,
        'tol': prox_infos[prox_name]['tol'],
        'verbose': True,
        'print_every': 300,
        'record_every': 300,
    }

    model_index, info = learn_one_model(model_file_path, strength_range,
                                        create_prox,
                                        compute_metrics, original_coeffs,
                                        SolverClass, solver_kwargs,
                                        save_coeffs=True)

    coeff_file_path = 'coeffs_{}_dim_{}_decays_{}_T_{}.npy'\
        .format(prox_name, dim, n_decays, end_time)

    with open(coeff_file_path, 'wb') as output_file:
        np.save(output_file, info['coeffs'])

    o_baseline_, o_adjacency_ = mus_alphas_from_coeffs(
        original_coeffs, n_decays_)
    plot_coeffs_3_decays(coeff_file_path, max_adjacency=o_adjacency_.max())


if __name__ == '__main__':
    dim_ = 30
    n_decays_ = 3
    end_time_ = 20000

    # prox_name_ = 'l1'
    # prox_name_ = 'l1w_un'
    prox_name_ = 'l1_nuclear'
    # prox_name_ = 'l1w_un_nuclear'

    learn_coeffs(dim_, n_decays_, end_time_, prox_name_)
