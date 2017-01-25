import numpy as np
from mlpp.base.utils import TimeFunction
from pylab import rcParams
from mlpp.simulation.inhomogeneous_poisson import SimuInhomogeneousPoisson

rcParams['figure.figsize'] = 15, 4
run_time = 30

T = np.arange((run_time * 0.9) * 5, dtype=float) / 5
Y = np.maximum(15 * np.sin(T) * (np.divide(np.ones_like(T),
                                           np.sqrt(T + 1) + 0.1 * T)), 0.001)

tf = TimeFunction((T, Y), dt=0.01)

# We define a 1 dimentional inhomogeneous Poisson process with the
# intensity function seen above
in_poi = SimuInhomogeneousPoisson([tf], end_time=run_time)

# We activate intensity tracking and launch the process during t = runtime
in_poi.track_intensity(0.1)
in_poi.simulate()

# We plot the resulting inhomogeneous Poisson process with its
# intensity and its ticks over time
in_poi.plot()
