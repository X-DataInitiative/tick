import os


def get_next_file_name(directory, filename, extension='npy'):
    os.makedirs(directory, exist_ok=True)
    existing_files = os.listdir(directory)

    suffix = 0
    txt_suffix = '%03d' % suffix  # '%03d' allow us to have leading zeros
    full_filename = '%s_%s.%s' % (filename, txt_suffix, extension)
    while full_filename in existing_files:
        suffix += 1
        txt_suffix = '%03d' % suffix  # '%03d' allow us to have leading zeros
        full_filename = '%s_%s.%s' % (filename, txt_suffix, extension)
    return full_filename


def get_coeffs_dir(dim, directory_prefix):
    train_directory = os.path.join(
        directory_prefix, 'train_hawkes/dim_{}/'.format(dim))
    return train_directory


def get_simulation_dir(dim, run_time, directory_prefix):
    train_directory_simulations = os.path.join(
        get_coeffs_dir(dim, directory_prefix),
        'T_{:.0f}/'.format(run_time))
    return train_directory_simulations


def get_precomputed_models_dir(dim, run_time, directory_prefix):
    train_directory_simulations = os.path.join(
        get_coeffs_dir(dim, directory_prefix),
        'T_{:.0f}/precomputed/'.format(run_time))
    return train_directory_simulations


def load_directory(directory, extension):
    if os.path.exists(directory):
        return [filename for filename in os.listdir(directory)
                if filename.endswith(extension)]
    else:
        return []


if __name__ == '__main__':
    print(get_coeffs_dir(10, 'here'))
    print(get_simulation_dir(10, 1500., 'here'))
