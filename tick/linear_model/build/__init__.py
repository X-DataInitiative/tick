# License: BSD 3 clause

# Templating update - 19/02/2018
#  Importing DLLs has gotten a bit strange and now requires
#  updating the path for DLLs to be found before hand

from tick.base.opsys import add_to_path_if_windows, load_extension, \
    resolve_repo_root

add_to_path_if_windows(__file__)
_REPO_ROOT = resolve_repo_root(__file__, levels_up=2)


def _load_extension(module_name):
    module = load_extension(module_name, __name__, __file__,
                            repo_root=_REPO_ROOT)
    globals()[module_name] = module
    return module


linear_model = _load_extension("linear_model")
