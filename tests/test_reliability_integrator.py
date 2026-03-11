import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pytest
from failure_paths.common.prob import beta_from_pf, pf_from_beta
from failure_paths.reliability import (
    FailureSamples,
    FragilityCurve,
    HazardCurve,
    IntegrationConfig,
    IntegrationResult,
    ReliabilityIntegrator,
)
from failure_paths.reliability.plotting import plot_integration_grid, prepare_failure_histogram


def _default_config(**overrides: float) -> IntegrationConfig:
    kwargs = {
        "r_distribution": ot.Gumbel(1.0, 4.0),
        "s_distribution": ot.Normal(1.0, 1.0),
        "coarse_points": 11,
        "u_min": -8.0,
        "u_max": 8.0,
    }
    kwargs.update(overrides)
    return IntegrationConfig(**kwargs)


def _form_reference_result(
    r_distribution: ot.Distribution, s_distribution: ot.Distribution, threshold: float = 0.0
) -> tuple[float, np.ndarray]:
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

    optim_algo = ot.AbdoRackwitz()
    optim_algo.setMaximumCallsNumber(10000)
    optim_algo.setMaximumAbsoluteError(1e-10)
    optim_algo.setMaximumRelativeError(1e-10)
    optim_algo.setMaximumResidualError(1e-10)
    optim_algo.setMaximumConstraintError(1e-10)
    optim_algo.setStartingPoint(distribution.getMean())

    form = ot.FORM(optim_algo, event)
    form.run()
    result = form.getResult()
    beta = float(result.getHasoferReliabilityIndex())
    alphas = -np.array(result.getStandardSpaceDesignPoint()) / beta

    algo = ot.PostAnalyticalImportanceSampling(result)
    algo.setMaximumCoefficientOfVariation(1e-4)
    algo.setBlockSize(int(1e5))
    algo.setMaximumOuterSampling(int(1e2))
    algo.run()
    result2 = algo.getResult()
    beta = ot.Normal().computeQuantile(result2.getProbabilityEstimate(), True)[0]

    return beta, alphas


def test_integrator_matches_analytic_pf() -> None:
    mu_r, sigma_r = 1.0, 1.0
    mu_s, sigma_s = 0.0, 0.5
    config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    sigma_z = math.hypot(sigma_r, sigma_s)
    # analytic_pf = ot.Normal(mu_r - mu_s, sigma_z).computeCDF(0.0)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - mu_s) / sigma_z

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=1e-4, abs_tol=0)


def test_integrator_matches_dirac_solicitation() -> None:
    mu_r, sigma_r = 2.0, 0.25
    s_level = 1.7
    config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Dirac(s_level),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()

    # analytic_pf = ot.Normal(mu_r, sigma_r).computeCDF(s_level)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - s_level) / sigma_r

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=0.0, abs_tol=1e-3)


@pytest.mark.parametrize("step", [3.1, 3.6, 4.2, 4.3, 5.2])
def test_integrator_matches_normal_step_solicitation(step: float) -> None:
    mu_s, sigma_s = 4.0, 0.25
    config = IntegrationConfig(
        s_distribution=ot.Normal(mu_s, sigma_s),
        fragility_curve=FragilityCurve([step - 1.0, step, step + 1e-6, step + 1], [np.inf, np.inf, -np.inf, -np.inf]),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()

    analytic_beta = ot.Normal(mu_s, sigma_s).computeSurvivalFunction([step])
    analytic_beta = ot.Normal().computeQuantile(analytic_beta, True)[0]

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=0.0, abs_tol=1e-3)


@pytest.mark.parametrize("step", [3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.4])
def test_integrator_matches_hazard_step_solicitation(step: float) -> None:
    hazard_data = [
        (3.6, 0.1081979),
        (3.7, 8.7795116e-02),
        (3.8, 7.0926525e-02),
        (3.9, 5.7202410e-02),
        (4.0, 4.6065349e-02),
        (4.1, 3.7077814e-02),
        (4.2, 2.9814601e-02),
        (4.3, 2.3914794e-02),
        (4.4, 1.9138306e-02),
        (4.5, 1.5251501e-02),
        (4.6, 1.2031388e-02),
        (4.7, 9.2987102e-03),
        (4.8, 6.9610281e-03),
        (4.9, 4.9987440e-03),
        (5.0, 3.4132395e-03),
        (5.1, 2.2145538e-03),
        (5.2, 1.3648920e-03),
        (5.3, 8.0153148e-04),
        (5.4, 4.5302085e-04),
        (5.5, 2.3970423e-04),
        (5.6, 1.2130349e-04),
        (5.7, 6.1366991e-05),
        (5.8, 3.1017240e-05),
        (5.9, 1.6166996e-05),
        (6.0, 8.8319548e-06),
        (6.1, 4.8836955e-06),
        (6.2, 2.7053347e-06),
        (6.3, 1.5057763e-06),
        (6.4, 8.2912675e-07),
    ]
    hazard_array = np.array(hazard_data, dtype=float)
    hazard_curve = HazardCurve(hazard_array[:, 0], hazard_array[:, 1])

    config = IntegrationConfig(
        hazard_curve=hazard_curve,
        fragility_curve=FragilityCurve([step - 1.0, step, step + 1e-6, step + 1], [np.inf, np.inf, -np.inf, -np.inf]),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()
    analytic_beta = float(hazard_curve.beta_from_hazard(step))

    assert result.adaptive_converged is True
    assert result.adaptive_estimated_logpf_error is not None
    assert result.adaptive_estimated_logpf_error <= config.adaptive_logpf_tol + 1e-12
    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=0.0, abs_tol=1e-3)


def test_integrator_respects_solicitation_cutoff() -> None:
    mu_r, sigma_r = 2.0, 0.25
    s_level = 1.7
    base_config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Dirac(s_level),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )

    base_result = ReliabilityIntegrator(config=base_config).run()

    base_payload = base_config.model_dump(exclude={"max_solicitation_level"})
    cutoff_result = ReliabilityIntegrator(
        config=IntegrationConfig(
            **base_payload,
            max_solicitation_level=s_level - 1e-6,
        )
    ).run()

    inclusive_result = ReliabilityIntegrator(
        config=IntegrationConfig(
            **base_payload,
            max_solicitation_level=s_level,
        )
    ).run()

    assert base_result.pf > 0.0
    assert cutoff_result.pf == 0.0
    assert math.isclose(inclusive_result.pf, base_result.pf, rel_tol=1e-12, abs_tol=0.0)


def test_failure_samples_with_cutoff_do_not_exceed_level() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(1.5, 0.6),
        s_distribution=ot.Normal(0.8, 0.7),
        coarse_points=61,
        u_min=-8.0,
        u_max=8.0,
        max_solicitation_level=0.5,
    )
    result = ReliabilityIntegrator(config=config).run()
    levels = result.failure_samples.solicitation_levels
    assert levels is not None
    assert np.all(levels <= config.max_solicitation_level + 1e-12)


def test_integrator_matches_dirac_resistance() -> None:
    mu_s, sigma_s = 0.5, 0.4
    r_level = 1.4
    config = IntegrationConfig(
        r_distribution=ot.Dirac(r_level),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        u_min=-10.0,
        u_max=10.0,
    )
    result = ReliabilityIntegrator(config=config).run()

    s_dist = ot.Normal(mu_s, sigma_s)
    analytic_pf = 1.0 - float(s_dist.computeCDF(r_level))
    analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (r_level - mu_s) / sigma_s

    assert math.isclose(result.beta_pf, analytic_beta, rel_tol=0.0, abs_tol=1e-3)


def test_failure_samples_diagnostics_match_weights() -> None:
    config = _default_config(coarse_points=41)
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


def test_integrator_runs_with_zero_adaptive_depth() -> None:
    config = _default_config(coarse_points=31, adaptive_max_depth=0)
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    assert math.isfinite(result.beta_star)
    assert result.alpha.shape == (2,)


def test_integrator_matches_form_reference() -> None:
    config = _default_config(coarse_points=101)
    integrator = ReliabilityIntegrator(config=config)
    near_result = integrator.run()

    beta_form, alpha_form = _form_reference_result(
        config.r_distribution,
        config.s_distribution,
        config.threshold,
    )

    assert math.isclose(near_result.beta_pf, beta_form, rel_tol=1e-3)
    assert math.isclose(near_result.alpha[0], alpha_form[0], rel_tol=2e-2)
    assert math.isclose(near_result.alpha[1], alpha_form[1], rel_tol=2e-2)


def test_hazard_curve_validation() -> None:
    with pytest.raises(ValueError):
        HazardCurve([1.0], [0.8])
    with pytest.raises(ValueError):
        HazardCurve([1.0, 0.5], [0.8, 0.2])
    with pytest.raises(ValueError):
        HazardCurve([1.0, 2.0], [0.2, 0.8])

    assert HazardCurve([1.0, 2.0], [0.9, 0.2])
    assert HazardCurve([1.0, 2.0], [0.5, 0.5])


def test_fragility_curve_validation_requires_sorted_inputs() -> None:
    with pytest.raises(ValueError):
        FragilityCurve([1.0], [0.5])
    with pytest.raises(ValueError):
        FragilityCurve([1.0, 0.5], [0.0, 1.0])


def test_hazard_integrator_matches_expected_pf_and_hazard_level() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0, 3.0], [0.99, 0.8, 0.3, 0.02])
    fragility = FragilityCurve([0.0, 1.0, 2.0, 3.0], [1.2, 0.5, -0.5, -2.0])

    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=101,
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
    hazard = HazardCurve([0.0, 1.0, 2.0], [0.95, 0.5, 0.05])
    fragility = FragilityCurve([0.0, 1.0, 2.0], [1.0, 0.0, -1.5])

    # Curve only for R, distribution for S
    config_fragility = IntegrationConfig(
        s_distribution=ot.Normal(0.0, 1.0),
        fragility_curve=fragility,
        coarse_points=61,
    )
    result_fragility = ReliabilityIntegrator(config=config_fragility).run()

    # Curve only for S, distribution for R
    config_hazard = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        hazard_curve=hazard,
        coarse_points=61,
    )
    result_hazard = ReliabilityIntegrator(config=config_hazard).run()

    assert 0.0 < result_fragility.pf < 1.0
    assert 0.0 < result_hazard.pf < 1.0
    assert result_hazard.hazard_level is not None
    assert result_fragility.hazard_level is None


def test_hazard_curve_beta_mapping_roundtrip() -> None:
    hazard = HazardCurve([0.0, 2.0, 4.0], [0.95, 0.5, 0.05])
    betas = hazard.beta_from_hazard([0.0, 4.0])
    reconstructed = hazard.hazard_from_beta(betas)
    assert np.allclose(reconstructed, [0.0, 4.0], atol=1e-6)


def test_curve_knots_are_injected_into_u_grid_edges() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0, 3.0], [0.98, 0.8, 0.2, 0.02])
    fragility = FragilityCurve([0.0, 1.0, 2.0, 3.0], [2.0, 0.5, -0.5, -2.0])

    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=5,
        u_min=-6.0,
        u_max=6.0,
    )
    integrator = ReliabilityIntegrator(config=config)
    grid = integrator._compute_distribution_grid()

    r_levels = fragility.hazard_levels.astype(float)
    r_cdf = np.array(integrator.r_distribution.computeCDF(r_levels[:, np.newaxis])).reshape(-1)
    expected_u1 = beta_from_pf(np.clip(r_cdf, 0.0, 1.0), tail="lower")
    expected_u1 = expected_u1[(expected_u1 > config.u_min) & (expected_u1 < config.u_max) & np.isfinite(expected_u1)]

    s_levels = hazard.hazard_levels.astype(float)
    s_cdf = np.array(integrator.s_distribution.computeCDF(s_levels[:, np.newaxis])).reshape(-1)
    expected_u2 = beta_from_pf(np.clip(s_cdf, 0.0, 1.0), tail="lower")
    expected_u2 = expected_u2[(expected_u2 > config.u_min) & (expected_u2 < config.u_max) & np.isfinite(expected_u2)]

    for u_val in np.unique(expected_u1):
        assert np.any(np.isclose(grid.u1_edges, u_val, rtol=0.0, atol=1e-12))

    for u_val in np.unique(expected_u2):
        assert np.any(np.isclose(grid.u2_edges, u_val, rtol=0.0, atol=1e-12))


def test_curve_knots_fill_regular_edges_up_to_coarse_points() -> None:
    u_min, u_max = -4.0, 4.0
    knot_betas = np.linspace(u_min, u_max, 80)
    hazard_levels = np.arange(knot_betas.size, dtype=float)
    exceedance = pf_from_beta(knot_betas, tail="upper")
    hazard = HazardCurve(hazard_levels, exceedance)

    cfg = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        hazard_curve=hazard,
        coarse_points=101,
        u_min=u_min,
        u_max=u_max,
    )
    grid = ReliabilityIntegrator(cfg)._compute_distribution_grid()

    assert grid.u2_edges.size == cfg.coarse_points
    assert np.any(np.isclose(grid.u2_edges, u_min, rtol=0.0, atol=1e-12))
    assert np.any(np.isclose(grid.u2_edges, u_max, rtol=0.0, atol=1e-12))


def test_curve_knots_are_not_downsampled_when_exceeding_coarse_points() -> None:
    u_min, u_max = -4.0, 4.0
    knot_betas = np.linspace(u_min, u_max, 160)
    hazard_levels = np.arange(knot_betas.size, dtype=float)
    exceedance = pf_from_beta(knot_betas, tail="upper")
    hazard = HazardCurve(hazard_levels, exceedance)

    cfg = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        hazard_curve=hazard,
        coarse_points=101,
        u_min=u_min,
        u_max=u_max,
    )
    integrator = ReliabilityIntegrator(cfg)
    grid = integrator._compute_distribution_grid()

    level_arr = hazard.hazard_levels.astype(float)
    cdf = np.array(integrator.s_distribution.computeCDF(level_arr[:, np.newaxis])).reshape(-1)
    knot_u = beta_from_pf(np.clip(cdf, 0.0, 1.0), tail="lower")
    knot_u = knot_u[np.isfinite(knot_u) & (knot_u > u_min) & (knot_u < u_max)]
    mandatory = np.sort(np.unique(np.r_[u_min, u_max, knot_u]))

    assert mandatory.size > cfg.coarse_points
    assert grid.u2_edges.size == mandatory.size
    assert np.allclose(grid.u2_edges, mandatory, rtol=0.0, atol=1e-12)


def test_hazard_fragility_from_normals_matches_distribution_result() -> None:
    mu_r, sigma_r = 2.5, 0.8
    mu_s, sigma_s = 0.5, 0.9
    s_factor = 10
    dist_config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Normal(mu_s, sigma_s),
        coarse_points=101,
        u_min=-s_factor,
        u_max=s_factor,
    )
    dist_result = ReliabilityIntegrator(config=dist_config).run()

    hazard_dist = ot.Normal(mu_s, sigma_s)
    fragility_dist = ot.Normal(mu_r, sigma_r)
    level_min = mu_s - s_factor * sigma_s
    level_max = mu_s + s_factor * sigma_s
    hazard_levels = np.linspace(level_min, level_max, 1010)
    exceedance_probs = np.array(hazard_dist.computeSurvivalFunction(hazard_levels[:, np.newaxis])).flatten()
    hazard_curve = HazardCurve(hazard_levels, exceedance_probs=exceedance_probs)

    level_min = mu_r - s_factor * sigma_r
    level_max = mu_r + s_factor * sigma_r
    hazard_levels = np.linspace(level_min, level_max, 1010)
    low_mask = hazard_levels <= mu_r
    cdf_vals = np.array(fragility_dist.computeCDF(hazard_levels[:, np.newaxis])).flatten()
    sur_vals = np.array(fragility_dist.computeSurvivalFunction(hazard_levels[:, np.newaxis])).flatten()
    beta_knots = np.empty_like(hazard_levels)
    beta_knots[low_mask] = np.array(ot.Normal().computeQuantile(cdf_vals[low_mask], True)).flatten()
    beta_knots[~low_mask] = np.array(ot.Normal().computeQuantile(sur_vals[~low_mask])).flatten()
    fragility_curve = FragilityCurve(hazard_levels, beta_knots)

    curve_config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard_curve,
        fragility_curve=fragility_curve,
        coarse_points=101,
    )
    curve_result = ReliabilityIntegrator(config=curve_config).run()
    analytic_beta = (mu_r - mu_s) / math.hypot(sigma_r, sigma_s)

    # Curve-driven grids now include curve knot locations, so they are not
    # expected to be bitwise-equivalent to distribution-only grids.
    assert math.isclose(curve_result.beta_pf, dist_result.beta_pf, rel_tol=2e-4)
    assert abs(curve_result.beta_pf - analytic_beta) <= abs(dist_result.beta_pf - analytic_beta) + 1e-12


def test_integration_grid_plot_smoke(tmp_path: Path) -> None:
    config = _default_config(coarse_points=21)
    integrator = ReliabilityIntegrator(config=config)
    fig, ax = plot_integration_grid(integrator)

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
    hazard = HazardCurve([0.0, 1.0, 2.0], [0.95, 0.4, 0.05])
    fragility = FragilityCurve([0.0, 1.0, 2.0], [1.0, 0.0, -1.5])
    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=21,
    )
    integrator = ReliabilityIntegrator(config=config)
    fig, ax = plot_integration_grid(integrator, limit_points=65)

    assert fig.axes and fig.axes[0] is ax
    output = tmp_path / "grid_hazard.png"
    fig.savefig(output)
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)


def test_failure_histogram_conserves_probability() -> None:
    config = _default_config(coarse_points=51)
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    water_levels = result.failure_water_levels()
    assert water_levels is not None

    level_min = float(water_levels.min()) - 1e-6
    level_max = float(water_levels.max()) + 1e-6
    edges = np.linspace(level_min, level_max, 32)

    hist = prepare_failure_histogram(result, edges)
    assert np.isclose(hist["failure_mass"].sum(), result.pf, rtol=1e-12, atol=1e-15)

    cond_hist = prepare_failure_histogram(
        result,
        edges,
        conditional=True,
        solicitation_distribution=config.s_distribution,
    )
    cdf_edges = np.array(config.s_distribution.computeCDF(cond_hist["bin_edges"][:, np.newaxis])).flatten()
    bin_probs = np.diff(cdf_edges)
    reconstructed_pf = float(np.sum(cond_hist["conditional_failure"] * bin_probs))
    assert np.isclose(reconstructed_pf, result.pf, rtol=1e-12, atol=1e-15)


def test_failure_histogram_handles_empty_samples() -> None:
    samples = FailureSamples(weights=np.array([]), points=np.empty((0, 2)))
    result = IntegrationResult(
        pf=0.0,
        beta_star=0.0,
        alpha=np.zeros(2),
        failure_samples=samples,
        beta_pf=0.0,
    )
    edges = np.array([0.0, 1.0, 2.0])

    hist = prepare_failure_histogram(result, edges)
    assert np.allclose(hist["failure_mass"], 0.0)

    cond_hist = prepare_failure_histogram(
        result,
        edges,
        conditional=True,
        solicitation_distribution=ot.Normal(),
    )
    assert np.allclose(cond_hist["conditional_failure"], 0.0)
