from tick.simulation import SimuPoissonProcess
from tick.plot import plot_point_process

run_time = 10
intensity = 5

poi = SimuPoissonProcess(intensity, end_time=run_time)
poi.simulate()
plot_point_process(poi)
