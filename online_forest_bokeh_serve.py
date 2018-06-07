import subprocess
import shlex

command = "bokeh serve --show online_forest_viz.py"

bokeh_serve = None

try:
    bokeh_serve = subprocess.Popen(shlex.split(command), shell=False,
                                   stdout=subprocess.PIPE)
except KeyboardInterrupt:
    bokeh_serve.kill()
