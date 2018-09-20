# License: BSD 3 clause


import datetime
import os
from subprocess import PIPE, run

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
