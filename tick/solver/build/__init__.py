# License: BSD 3 clause

# Templating update - 19/02/2018
#  Importing DLLs has gotten a bit strange and now requires
#  updating the path for DLLs to be found before hand
#  As observed here both "base_model" and "linear_model" are required

from tick.base.opsys import add_to_path_if_windows


def required():
    import os, sys
    root = os.path.dirname(os.path.realpath(os.path.join(__file__, "../..")))

    deps = ["base_model", "linear_model", "robust"]

    for dep in deps:
        if "tick." + dep + ".build" not in sys.modules:
            p = os.path.realpath(os.path.join(root, dep + "/build"))
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]


add_to_path_if_windows(__file__, [required])
