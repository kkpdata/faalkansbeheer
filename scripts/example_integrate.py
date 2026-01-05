from pprint import pprint

import openturns as ot
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.plotting import IntegrationGridPlotter
from matplotlib import pyplot as plt

config = IntegrationConfig(
    r_distribution=ot.Gumbel(1.0, 4.0),
    s_distribution=ot.Normal(1.0, 1.0),
    coarse_points=11,
    refine_factor=10,
    u_min=-8.0,
    u_max=8.0,
)
integrator = ReliabilityIntegrator(config=config)
result = integrator.run()

# show summary
pprint(result.summary())

# visualize
plotter = IntegrationGridPlotter(integrator)
fig, ax = plotter.plot()
plt.show()
