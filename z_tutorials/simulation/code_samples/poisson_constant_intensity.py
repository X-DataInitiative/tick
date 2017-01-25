from pylab import rcParams
from mlpp.simulation import SimuPoissonProcess

rcParams['figure.figsize'] = 10, 4
run_time = 10000
intensity = 5

# We define a 1 dimensional homogeneous Poisson process
poi = SimuPoissonProcess(intensity, end_time=run_time)

# We launch the process during t = runtime
poi.simulate()
poi.plot(style='hist')
