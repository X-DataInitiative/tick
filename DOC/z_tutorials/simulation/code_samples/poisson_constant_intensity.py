import matplotlib.pyplot as plt
from tick.simulation import SimuPoissonProcess
from tick.plot import plot_point_process

run_time = 10
intensity = 5

# We define a 1 dimensional homogeneous Poisson process
poi = SimuPoissonProcess(intensity, end_time=run_time)

# We launch the process during t = runtime
poi.simulate()

plot_point_process(poi)
