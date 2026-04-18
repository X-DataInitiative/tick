# License: BSD 3 clause

from tick.base.opsys import add_to_path_if_windows, load_extension, \
    resolve_repo_root

add_to_path_if_windows(__file__)
_REPO_ROOT = resolve_repo_root(__file__, levels_up=2)


def _load_extension(module_name):
    module = load_extension(module_name, __name__, __file__,
                            repo_root=_REPO_ROOT)
    globals()[module_name] = module
    return module


preprocessing = _load_extension("preprocessing")
