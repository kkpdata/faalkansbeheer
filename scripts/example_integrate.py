from pprint import pprint

import numpy as np
import openturns as ot
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_diagnostics_1d
from matplotlib import pyplot as plt

config = IntegrationConfig(
    r_distribution=ot.Gumbel(1.0, 4.0),
    s_distribution=ot.Normal(1.0, 1.0),
    coarse_points=501,
    max_solicitation_level=4,
)
integrator = ReliabilityIntegrator(config=config)
result = integrator.run(collect_diagnostics_trace=True)

# show summary
pprint(result.summary())

# visualize
fig, axs = plt.subplots(ncols=3, figsize=(18, 5), dpi=100)
diag_axes = np.array([axs[0], axs[1]], dtype=object)
hist_ax = axs[2]
plot_integration_diagnostics_1d(result, axes=diag_axes)

# visualize failure probability distribution over water levels
water_levels = np.linspace(0, 10, 101)
plot_failure_histogram(result, water_levels, ax=hist_ax)

plt.show()
