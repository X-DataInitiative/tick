# License: BSD 3 clause

"""Import shim for :mod:`tick.hawkes.inference.build`."""

from tick.base.opsys import add_to_path_if_windows, load_extension, \
    resolve_repo_root

add_to_path_if_windows(__file__)
_REPO_ROOT = resolve_repo_root(__file__, levels_up=3)


def _load_hawkes_inference_module():
    return load_extension("hawkes_inference", __name__, __file__,
                          repo_root=_REPO_ROOT)


_hawkes_inference = _load_hawkes_inference_module()
__all__ = [name for name in getattr(_hawkes_inference, "__all__",
                                    dir(_hawkes_inference))
           if not name.startswith("_")]

for _name in __all__:
    globals()[_name] = getattr(_hawkes_inference, _name)

del _hawkes_inference
