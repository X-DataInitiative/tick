# License: BSD 3 clause
"""This file provides utilities to easily launch the various benchmarks that
are provided to test tick code
"""

import datetime
import os
from subprocess import PIPE, run

from tick.array.serialize import serialize_array
from tick.dataset import fetch_tick_dataset
from tick.dataset.fetch_url_dataset import fetch_url_dataset

BUILD_SOURCE_DIR = os.path.join(
    os.path.dirname(__file__), '../../', 'build/bench')

builds = ['build_noopt', 'build_mkl', 'build_blas']


def iter_executables(*executables):
    for build in builds:
        if not os.path.exists(os.path.join(BUILD_SOURCE_DIR, build)):
            print('missing', build)
            continue
        for executable in executables:
            yield os.path.join(BUILD_SOURCE_DIR, build,
                               'out', executable)


def extract_build_from_name(name):
    for build in builds:
        if build in name:
            return build

    raise ValueError('Cannot find build in {}'.format(name))


def get_default_result_filename(executable_path):
    result_file_name = executable_path
    result_file_name = result_file_name.replace(BUILD_SOURCE_DIR, '')
    result_file_name = result_file_name.replace('out/', '')
    result_file_name = result_file_name.strip('/')
    result_file_name = result_file_name.replace('/', '_')
    result_file_name += '.tsv'
    return result_file_name


def default_result_dir(base=''):
    return os.path.join(os.path.dirname(__file__), 'results', base,
                        '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))


def get_last_result_dir(base=''):
    bench_result_dir = default_result_dir(base=base)
    bench_result_dir = os.path.dirname(bench_result_dir)
    result_dirs = os.listdir(bench_result_dir)
    result_dirs.sort()
    return os.path.join(bench_result_dir, result_dirs[-1])


def run_benchmark(executable, ex_args, output_dir):
    result_file_path = os.path.join(output_dir,
                                    get_default_result_filename(executable))

    result_dir = os.path.dirname(result_file_path)
    os.makedirs(result_dir, exist_ok=True)
    print("result_file_path , ", result_file_path)
    with open(result_file_path, 'w') as output_file:
        print("Writing to", result_file_path)

        for args in ex_args:
            if args:
                command = [executable, str(args)]
            else:
                command = executable
            result = run(command, stdout=PIPE, stderr=PIPE,
                         universal_newlines=True)

            output_file.write(result.stdout)

            if result.stderr:
                print('Failed', result.stderr)

        output_file.flush()
    print(executable, "Done")

    return result_dir


def save_url_dataset_for_cpp_benchmarks(n_days):
    """Fetches and saves as C++ cereal serialized file the URL dataset

    Parameters
    ----------
    n_days : `int`
        Number of days kept from the original dataset.
        As this dataset is quite big, you might not want to use it in totality.
    """
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '../../tools/benchmark/data')

    label_path = os.path.join(save_path, 'url.{}.labels.cereal'.format(n_days))
    features_path = os.path.join(save_path,
                                 'url.{}.features.cereal'.format(n_days))

    X, y = fetch_url_dataset(n_days=n_days)
    serialize_array(y, label_path)
    serialize_array(X, features_path)


def save_adult_dataset_for_cpp_benchmarks():
    """Fetches and saves as C++ cereal serialized file the adult dataset
    """
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '../../tools/benchmark/data')

    label_path = os.path.join(save_path, 'adult.labels.cereal')
    features_path = os.path.join(save_path, 'adult.features.cereal')

    X, y = fetch_tick_dataset('binary/adult/adult.trn.bz2')
    serialize_array(y, label_path)
    serialize_array(X, features_path)
