# License: BSD 3 clause

# Templating update - 19/02/2018
#  Importing DLLs has gotten a bit strange and now requires
#  updating the path for DLLs to be found before hand
#  As observed here "base_model" are required

from tick.base.opsys import add_to_path_if_windows


def required():
    import os, sys
    root = os.path.dirname(
        os.path.realpath(os.path.join(__file__, "../../..")))

    if "tick.base_model.build" not in sys.modules:
        p = os.path.realpath(os.path.join(root, "base_model/build"))
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]


add_to_path_if_windows(__file__, [required])
