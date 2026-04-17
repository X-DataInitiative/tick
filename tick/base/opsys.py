# This file exists to allow for different functionality
#  between operating systems if so when required

import importlib
import importlib.machinery
import importlib.util
import os
import platform
import sys
from pathlib import Path


# This function will add all the module build directories
#  to the system path if the sysem is deemed Windows
def add_to_path_if_windows(file, funcs=None):
    if funcs is None:
        funcs = []

    if platform.system() == 'Windows':
        for func in funcs:
            func()
        dir = os.path.dirname(os.path.realpath(file))
        if dir not in os.environ["PATH"]:
            os.environ["PATH"] = dir + os.pathsep + os.environ["PATH"]


def resolve_repo_root(package_file, *, levels_up=None, marker=None):
    package_dir = Path(package_file).resolve().parent

    if marker is not None:
        return next(parent for parent in [package_dir, *package_dir.parents]
                    if (parent / marker).exists())

    if levels_up is None:
        raise ValueError("Either levels_up or marker must be provided")

    return package_dir.parents[levels_up]


def _iter_extension_candidates(module_name, package_dir, repo_root,
                               search_roots):
    try:
        build_dir_parts = package_dir.relative_to(repo_root).parts
    except ValueError:
        return
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)

    for search_root in search_roots:
        build_root = repo_root if search_root is None else repo_root / search_root
        if build_root != repo_root and not build_root.exists():
            continue

        for candidate in sorted(build_root.rglob(f"{module_name}*")):
            if not candidate.is_file():
                continue

            if build_root != repo_root:
                relative_candidate = candidate.relative_to(build_root)
                if relative_candidate.parts[1:1 + len(build_dir_parts)] != build_dir_parts:
                    continue

            if candidate.name[len(module_name):] not in extension_suffixes:
                continue

            yield candidate


def load_extension(module_name, package_name, package_file, *, repo_root,
                   search_roots=("_skbuild", "build")):
    qualified_name = f"{package_name}.{module_name}"
    package_dir = Path(package_file).resolve().parent

    try:
        return importlib.import_module(f".{module_name}", package_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {qualified_name, module_name}:
            raise

    for candidate in _iter_extension_candidates(module_name, package_dir,
                                                repo_root, search_roots):
        spec = importlib.util.spec_from_file_location(qualified_name,
                                                      candidate)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(qualified_name)
        sys.modules[qualified_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if previous_module is None:
                sys.modules.pop(qualified_name, None)
            else:
                sys.modules[qualified_name] = previous_module
            raise
        return module

    raise ModuleNotFoundError(
        f"No module named '{qualified_name}'", name=qualified_name)
