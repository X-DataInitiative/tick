import re
import subprocess
import sys
import warnings

python_min_ver = (3, 4, 0)
swig_min_ver = (3, 0, 7)


def get_swig_version(swig_ver_str):
    m = re.search('SWIG Version (\d+).(\d+).(\d+)', swig_ver_str)

    if not m:
        warnings.warn(
            'Could not extract SWIG version from string: {0}'
                .format(swig_ver_str))

        return 0, 0, 0

    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compare_versions(actual, expected):
    return actual >= expected


if __name__ == '__main__':
    python_ver = sys.version_info.major, \
                 sys.version_info.minor, \
                 sys.version_info.micro

    swig_ver = get_swig_version(
        str(subprocess.check_output(['swig', '-version'])))

    warned = False

    if not compare_versions(python_ver, python_min_ver):
        txt = 'Warning: Python version {0}.{1}.{2} ' \
              'lower than the required version >= {3}.{4}.{5}.'

        print(txt.format(*(python_ver + python_min_ver)))

        warned = True

    if not compare_versions(swig_ver, swig_min_ver):
        txt = 'Warning: SWIG version {0}.{1}.{2} ' \
              'lower than the required version >= {3}.{4}.{5}. ' \
              'This will likely cause build errors!'

        print(txt.format(*(swig_ver + swig_min_ver)))

        warned = True

    if warned:
        sys.exit(1)
