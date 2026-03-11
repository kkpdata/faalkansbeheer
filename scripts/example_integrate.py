from pprint import pprint

import numpy as np
import openturns as ot
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_grid
from matplotlib import pyplot as plt

config = IntegrationConfig(
    r_distribution=ot.Gumbel(1.0, 4.0),
    s_distribution=ot.Normal(1.0, 1.0),
    coarse_points=501,
    u_min=-8.0,
    u_max=8.0,
    max_solicitation_level=4,
)
integrator = ReliabilityIntegrator(config=config)
result = integrator.run()

# show summary
pprint(result.summary())

# visualize
fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=100)
plot_integration_grid(integrator, ax=axs[0])

# visualize failure probability distribution over water levels
water_levels = np.linspace(0, 10, 101)
plot_failure_histogram(result, water_levels, ax=axs[1])

plt.show()

