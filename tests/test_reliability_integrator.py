import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pytest
from failure_paths.reliability import (
    FragilityCurve,
    HazardCurve,
    IntegrationConfig,
    ReliabilityIntegrator,
)
from failure_paths.reliability.plotting import IntegrationGridPlotter


def _default_config(**overrides: float) -> IntegrationConfig:
    kwargs = {
        "r_distribution": ot.Gumbel(1.0, 4.0),
        "s_distribution": ot.Normal(1.0, 1.0),
        "coarse_points": 11,
        "refine_factor": 1,
        "u_min": -8.0,
        "u_max": 8.0,
    }
    kwargs.update(overrides)
    return IntegrationConfig(**kwargs)


def _form_reference_result(
    r_distribution: ot.Distribution, s_distribution: ot.Distribution, threshold: float = 0.0
) -> tuple[float, float]:
    marginals = {"R": r_distribution, "S": s_distribution}
    distribution = ot.ComposedDistribution(
        list(marginals.values()),
        ot.IndependentCopula(len(marginals)),
    )
    distribution.setDescription(list(marginals.keys()))

    g_function = ot.SymbolicFunction(["R", "S"], ["R - S"])
    random_vector = ot.RandomVector(distribution)
    composite = ot.CompositeRandomVector(g_function, random_vector)
    event = ot.ThresholdEvent(composite, ot.Less(), threshold)
    event.setName("failure")

    optim_algo = ot.Cobyla()
    optim_algo.setMaximumCallsNumber(10000)
    optim_algo.setMaximumAbsoluteError(1e-10)
    optim_algo.setMaximumRelativeError(1e-10)
    optim_algo.setMaximumResidualError(1e-10)
    optim_algo.setMaximumConstraintError(1e-10)
    optim_algo.setStartingPoint(distribution.getMean())

    form = ot.FORM(optim_algo, event)
    form.run()
    result = form.getResult()
    return float(result.getEventProbability()), float(result.getHasoferReliabilityIndex())


def test_integrator_matches_analytic_pf() -> None:
    mu_r, sigma_r = 1.0, 1.0
    mu_s, sigma_s = 0.0, 0.5
    config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        refine_factor=20,
        u_min=-10.0,
        u_max=10.0,
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    sigma_z = math.hypot(sigma_r, sigma_s)
    # analytic_pf = ot.Normal(mu_r - mu_s, sigma_z).computeCDF(0.0)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - mu_s) / sigma_z

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=1e-5, abs_tol=0)


def test_integrator_matches_dirac_solicitation() -> None:
    mu_r, sigma_r = 2.0, 0.25
    s_level = 1.7
    config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Dirac(s_level),
        coarse_points=101,
        refine_factor=20,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()

    # analytic_pf = ot.Normal(mu_r, sigma_r).computeCDF(s_level)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - s_level) / sigma_r

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=1e-12, abs_tol=0)


def test_integrator_matches_dirac_resistance() -> None:
    mu_s, sigma_s = 0.5, 0.4
    r_level = 1.4
    config = IntegrationConfig(
        r_distribution=ot.Dirac(r_level),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        refine_factor=20,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()

    s_dist = ot.Normal(mu_s, sigma_s)
    analytic_pf = 1.0 - float(s_dist.computeCDF(r_level))
    analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (r_level - mu_s) / sigma_s

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=1e-12, abs_tol=0)


def test_failure_samples_diagnostics_match_weights() -> None:
    config = _default_config(coarse_points=41, refine_factor=4)
    integrator = ReliabilityIntegrator(config=config)
    samples = integrator.integrate_failure_samples()

    assert samples.coarse_fail_cells >= 0
    assert samples.mixed_cells >= 0
    assert samples.refined_fail_cells >= 0
    assert samples.weights.ndim == 1
    assert samples.points.shape[1] == 2

    pf_from_samples = float(samples.weights.sum())
    result_pf = integrator.run().pf
    assert math.isclose(pf_from_samples, result_pf, rel_tol=1e-12, abs_tol=0.0)


def test_config_range_validation() -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(
            r_distribution=ot.Normal(0.0, 1.0),
            s_distribution=ot.Normal(0.0, 1.0),
            u_min=1.0,
            u_max=0.0,
        )


def test_integrator_refine_factor_one() -> None:
    config = _default_config(refine_factor=1, coarse_points=31)
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    assert math.isfinite(result.beta_star)
    assert result.alpha.shape == (2,)


def test_integrator_matches_form_reference() -> None:
    config = _default_config(coarse_points=101, refine_factor=10)
    integrator = ReliabilityIntegrator(config=config)
    near_result = integrator.run()

    pf_form, beta_form = _form_reference_result(
        config.r_distribution,
        config.s_distribution,
        config.threshold,
    )

    assert math.isclose(near_result.beta_pf, beta_form, rel_tol=2e-2)


def test_hazard_curve_validation() -> None:
    with pytest.raises(ValueError):
        HazardCurve([1.0], [0.1])
    with pytest.raises(ValueError):
        HazardCurve([1.0, 0.5], [0.2, 0.8])
    with pytest.raises(ValueError):
        HazardCurve([1.0, 2.0], [0.8, 0.2])

    assert HazardCurve([1.0, 2.0], [0.2, 0.2])
    assert HazardCurve([1.0, 2.0], [0.1, 0.9])


def test_fragility_curve_validation_requires_matching_hazard() -> None:
    hazard = HazardCurve([1.0, 2.0], [0.1, 0.9])
    with pytest.raises(ValueError):
        FragilityCurve([1.0], [0.5], hazard)


def test_hazard_integrator_matches_expected_pf_and_hazard_level() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0, 3.0], [0.01, 0.1, 0.9, 0.99])
    fragility = FragilityCurve([0.0, 1.0, 2.0, 3.0], [0.0, 0.2, 0.8, 1.0], hazard)

    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=101,
        refine_factor=20,
        u_min=-10.0,
        u_max=10.0,
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()
    samples = result.failure_samples

    assert 0.0 < result.pf < 1.0
    assert result.hazard_level is not None
    assert samples.hazard_levels is not None
    assert samples.hazard_levels.shape[0] == samples.weights.shape[0]


def test_mixed_curve_and_distribution_inputs() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0], [0.05, 0.4, 0.95])
    fragility = FragilityCurve([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], hazard)

    # Curve only for R, distribution for S
    config_fragility = IntegrationConfig(
        s_distribution=ot.Normal(0.0, 1.0),
        fragility_curve=fragility,
        coarse_points=61,
        refine_factor=2,
    )
    result_fragility = ReliabilityIntegrator(config=config_fragility).run()

    # Curve only for S, distribution for R
    config_hazard = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        hazard_curve=hazard,
        coarse_points=61,
        refine_factor=2,
    )
    result_hazard = ReliabilityIntegrator(config=config_hazard).run()

    assert 0.0 < result_fragility.pf < 1.0
    assert 0.0 < result_hazard.pf < 1.0
    assert result_hazard.hazard_level is not None
    assert result_fragility.hazard_level is None


def test_hazard_curve_beta_mapping_roundtrip() -> None:
    hazard = HazardCurve([0.0, 2.0, 4.0], [0.05, 0.5, 0.95])
    betas = hazard.beta_from_hazard([0.0, 4.0])
    reconstructed = hazard.hazard_from_beta(betas)
    assert np.allclose(reconstructed, [0.0, 4.0], atol=1e-6)


def test_hazard_fragility_from_normals_matches_distribution_result() -> None:
    mu_r, sigma_r = 2.5, 0.8
    mu_s, sigma_s = 0.5, 0.9
    dist_config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        refine_factor=10,
    )
    dist_result = ReliabilityIntegrator(config=dist_config).run()

    hazard_dist = ot.Normal(mu_s, sigma_s)
    fragility_dist = ot.Normal(mu_r, sigma_r)
    level_min = mu_s - 8 * sigma_s
    level_max = mu_s + 8 * sigma_s
    hazard_levels = np.linspace(level_min, level_max, 2001)
    hazard_probs = np.array(hazard_dist.computeCDF(hazard_levels[:, np.newaxis])).flatten()
    hazard_curve = HazardCurve(hazard_levels, hazard_probs)
    fragility_probs = np.array(fragility_dist.computeCDF(hazard_levels[:, np.newaxis])).flatten()
    fragility_curve = FragilityCurve(hazard_levels, fragility_probs, hazard_curve)

    curve_config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard_curve,
        fragility_curve=fragility_curve,
        coarse_points=101,
        refine_factor=10,
    )
    curve_result = ReliabilityIntegrator(config=curve_config).run()

    assert math.isclose(curve_result.beta_pf, dist_result.beta_pf, rel_tol=1e-10)
    assert math.isclose(curve_result.alpha[0], dist_result.alpha[0], rel_tol=1e-10)
    assert math.isclose(curve_result.alpha[1], dist_result.alpha[1], rel_tol=1e-10)


def test_integration_grid_plot_smoke(tmp_path: Path) -> None:
    config = _default_config(coarse_points=21, refine_factor=4)
    integrator = ReliabilityIntegrator(config=config)
    plotter = IntegrationGridPlotter(integrator)
    fig, ax = plotter.plot()

    assert fig.axes and fig.axes[0] is ax
    assert ax.get_xlabel() == "$u_R$"
    assert ax.get_ylabel() == "$u_S$"

    labels = [line.get_label() for line in ax.get_lines()]
    assert "z = 0" in labels

    output = tmp_path / "grid.png"
    fig.savefig(output, bbox_inches="tight")
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)


def test_integration_grid_plot_with_hazard_curves(tmp_path: Path) -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0], [0.05, 0.4, 0.95])
    fragility = FragilityCurve([0.0, 1.0, 2.0], [0.0, 0.5, 1.0], hazard)
    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=21,
        refine_factor=2,
    )
    integrator = ReliabilityIntegrator(config=config)
    plotter = IntegrationGridPlotter(integrator)
    fig, ax = plotter.plot(limit_points=65)

    assert fig.axes and fig.axes[0] is ax
    output = tmp_path / "grid_hazard.png"
    fig.savefig(output)
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)
