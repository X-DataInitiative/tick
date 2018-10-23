import os
from itertools import product
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np

from experiments.hawkes_coeffs import retrieve_coeffs
from experiments.io_utils import get_next_file_name, get_simulation_dir, \
    load_directory
from tick.hawkes import SimuHawkesExpKernels


def simulate_and_save_hawkes(mu0, A0, betas, directory_path, run_time,
                             prefix='simulation'):
    hawkes = SimuHawkesExpKernels(A0, betas, baseline=mu0,
                                  end_time=run_time, verbose=False)
    hawkes.simulate()
    simulation = np.array(hawkes.timestamps)

    dim = len(mu0)
    directory = get_simulation_dir(dim, run_time, directory_path)
    filename = get_next_file_name(directory, prefix)
    file_path = os.path.join(directory, filename)
    np.save(file_path, simulation)
    return file_path


def simulate_hawkes_in_parallel(dim, run_times, n_simulations, directory_path,
                                n_cpu=-1):
    betas, mu0, A0 = retrieve_coeffs(dim, directory_path)

    if n_cpu < 1:
        n_cpu = cpu_count()

    args_list = [
        (mu0, A0, betas, directory_path, run_time)
        for _, run_time in product(range(n_simulations), run_times)]

    pool_sim = Pool(n_cpu)
    file_paths = pool_sim.starmap(simulate_and_save_hawkes, args_list)
    pool_sim.close()
    pool_sim.join()

    print('\n'.join(file_paths))


def get_simulation_files(dim, run_time, directory_prefix):
    directory = get_simulation_dir(dim, run_time, directory_prefix)
    return load_directory(directory, 'npy')


if __name__ == '__main__':
    dim = 30
    run_times = [500, 1000]

    n_simulations = 5
    directory_path = '/Users/martin/Downloads/jmlr_hawkes_data/'
    simulate_hawkes_in_parallel(dim, run_times, n_simulations, directory_path)
