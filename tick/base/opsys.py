# This file exists to allow for different functionality
#  between operating systems if so when required


# This function will add all the module build directories
#  to the system path if the sysem is deemed Windows
def add_to_path_if_windows(file, funcs=list()):
    import platform
    if platform.system() == 'Windows':
        import os
        for func in funcs:
            func()
        dir = os.path.dirname(os.path.realpath(file))
        if dir not in os.environ["PATH"]:
            os.environ["PATH"] = dir + os.pathsep + os.environ["PATH"]
