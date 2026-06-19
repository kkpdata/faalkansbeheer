import math
import warnings

import matplotlib

matplotlib.use("Agg")

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
from failure_paths.reliability.plotting import prepare_failure_histogram
from pydantic import ValidationError

ANALYTIC_BETA_REL_TOL = 1e-6
DIRAC_BETA_ABS_TOL = 1e-12
NORMAL_STEP_BETA_REL_TOL = 1e-11
HAZARD_STEP_BETA_REL_TOL = 1e-12
FORM_REFERENCE_SEED = 12345
FORM_BETA_REL_TOL = 1e-4
FORM_ALPHA_REL_TOL = 1e-4
CURVE_VS_DIST_BETA_REL_TOL = 1e-6
ZERO_TRUNCATION_PF_ERR_ATOL = 1e-14
GENERIC_REL_FALLBACK_TOL = 1e-12
GENERIC_BETA_ABS_FALLBACK_TOL = 1e-12
GENERIC_PF_ABS_FALLBACK_TOL = 1e-15


def _default_config(**overrides: object) -> IntegrationConfig:
    kwargs = {
        "r_distribution": ot.Gumbel(1.0, 4.0),
        "s_distribution": ot.Normal(1.0, 1.0),
        "coarse_points": 11,
    }
    kwargs.update(overrides)
    return IntegrationConfig(**kwargs)


def _form_reference_result(
    r_distribution: ot.Distribution,
    s_distribution: ot.Distribution,
) -> tuple[float, np.ndarray]:
    marginals = {"R": r_distribution, "S": s_distribution}
    distribution = ot.JointDistribution(
        list(marginals.values()),
        ot.IndependentCopula(len(marginals)),
    )
    distribution.setDescription(list(marginals.keys()))

    g_function = ot.SymbolicFunction(["R", "S"], ["R - S"])
    random_vector = ot.RandomVector(distribution)
    composite = ot.CompositeRandomVector(g_function, random_vector)
    event = ot.ThresholdEvent(composite, ot.Less(), 0.0)
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
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    sigma_z = math.hypot(sigma_r, sigma_s)
    # analytic_pf = ot.Normal(mu_r - mu_s, sigma_z).computeCDF(0.0)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - mu_s) / sigma_z

    assert result.converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound <= ZERO_TRUNCATION_PF_ERR_ATOL
    assert math.isclose(
        result.beta_pf,
        analytic_beta,
        rel_tol=ANALYTIC_BETA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )


def test_integrator_matches_dirac_solicitation() -> None:
    mu_r, sigma_r = 2.0, 0.25
    s_level = 1.7
    config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Dirac(s_level),
        coarse_points=101,
    )
    result = ReliabilityIntegrator(config=config).run()

    # analytic_pf = ot.Normal(mu_r, sigma_r).computeCDF(s_level)
    # analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (mu_r - s_level) / sigma_r

    assert result.converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound <= ZERO_TRUNCATION_PF_ERR_ATOL
    assert math.isclose(
        result.beta_pf,
        analytic_beta,
        rel_tol=GENERIC_REL_FALLBACK_TOL,
        abs_tol=DIRAC_BETA_ABS_TOL,
    )


@pytest.mark.parametrize("step", [3.1, 3.6, 4.2, 4.3, 5.2])
def test_integrator_matches_normal_step_solicitation(step: float) -> None:
    mu_s, sigma_s = 4.0, 0.25
    config = IntegrationConfig(
        s_distribution=ot.Normal(mu_s, sigma_s),
        fragility_curve=FragilityCurve([step - 1.0, step, step + 1e-6, step + 1], [np.inf, np.inf, -np.inf, -np.inf]),
        coarse_points=101,
    )
    result = ReliabilityIntegrator(config=config).run()

    analytic_beta = ot.Normal(mu_s, sigma_s).computeSurvivalFunction([step])
    analytic_beta = ot.Normal().computeQuantile(analytic_beta, True)[0]

    assert result.converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound <= ZERO_TRUNCATION_PF_ERR_ATOL
    assert math.isclose(
        result.beta_pf,
        analytic_beta,
        rel_tol=NORMAL_STEP_BETA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )


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
    )
    result = ReliabilityIntegrator(config=config).run()
    analytic_beta = float(hazard_curve.beta_from_hazard(step))

    assert result.adaptive_converged is True
    assert result.adaptive_estimated_logpf_error is not None
    assert result.adaptive_estimated_logpf_error <= config.adaptive_logpf_tol + 1e-12
    assert result.converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound <= ZERO_TRUNCATION_PF_ERR_ATOL
    assert math.isclose(
        result.beta_pf,
        analytic_beta,
        rel_tol=HAZARD_STEP_BETA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )


def test_integrator_respects_solicitation_cutoff() -> None:
    mu_r, sigma_r = 2.0, 0.25
    s_level = 1.7
    base_config = IntegrationConfig(
        r_distribution=ot.Normal(mu_r, sigma_r),
        s_distribution=ot.Dirac(s_level),
        coarse_points=101,
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
    assert math.isclose(
        inclusive_result.pf,
        base_result.pf,
        rel_tol=1e-12,
        abs_tol=GENERIC_PF_ABS_FALLBACK_TOL,
    )


def test_failure_samples_with_cutoff_do_not_exceed_level() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(1.5, 0.6),
        s_distribution=ot.Normal(0.8, 0.7),
        coarse_points=61,
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
    )
    result = ReliabilityIntegrator(config=config).run()

    s_dist = ot.Normal(mu_s, sigma_s)
    analytic_pf = 1.0 - float(s_dist.computeCDF(r_level))
    analytic_beta = ot.Normal().computeQuantile(analytic_pf, True)[0]
    analytic_beta = (r_level - mu_s) / sigma_s

    assert result.converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound <= ZERO_TRUNCATION_PF_ERR_ATOL
    assert math.isclose(
        result.beta_pf,
        analytic_beta,
        rel_tol=GENERIC_REL_FALLBACK_TOL,
        abs_tol=DIRAC_BETA_ABS_TOL,
    )


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
    assert math.isclose(pf_from_samples, result_pf, rel_tol=1e-12, abs_tol=GENERIC_PF_ABS_FALLBACK_TOL)


def test_grid_cell_weights_are_not_renormalized() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        s_distribution=ot.Normal(0.0, 1.0),
        coarse_points=101,
        u_manual_bounds=(-4.0, 4.0),
    )
    grid = ReliabilityIntegrator(config=config)._compute_distribution_grid()

    assert 0.0 < grid.captured_mass < 1.0
    assert np.isclose(grid.interval_probs.sum(), grid.captured_mass, rtol=0.0, atol=1e-15)


def test_discrete_support_levels_from_r_are_injected_into_s_u_edges() -> None:
    r_level = 1.7
    cfg = IntegrationConfig(
        r_distribution=ot.Dirac(r_level),
        s_distribution=ot.Normal(2.0, 0.25),
        coarse_points=101,
    )
    integrator = ReliabilityIntegrator(cfg)
    grid = integrator._compute_distribution_grid()

    cdf = float(np.clip(integrator.s_distribution.computeCDF(r_level), 0.0, 1.0))
    expected_u = float(beta_from_pf(cdf, tail="lower"))
    assert np.isfinite(expected_u)
    assert np.any(np.isclose(grid.u2_edges, expected_u, rtol=0.0, atol=1e-12))


def test_distribution_singularities_on_s_are_injected_into_u_edges() -> None:
    singular_level = 0.3
    mixture = ot.Mixture([ot.Dirac(singular_level), ot.Normal(0.0, 1.0)], [0.2, 0.8])
    cfg = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        s_distribution=mixture,
        coarse_points=101,
    )
    integrator = ReliabilityIntegrator(cfg)
    grid = integrator._compute_distribution_grid()

    cdf = float(np.clip(integrator.s_distribution.computeCDF(singular_level), 0.0, 1.0))
    expected_u = float(beta_from_pf(cdf, tail="lower"))
    assert np.isfinite(expected_u)
    assert np.any(np.isclose(grid.u2_edges, expected_u, rtol=0.0, atol=1e-12))


def test_manual_bounds_can_prevent_global_convergence_via_truncation() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(2.0, 1.0),
        s_distribution=ot.Normal(0.0, 1.0),
        coarse_points=81,
        u_manual_bounds=(-1.0, 1.0),
    )
    result = ReliabilityIntegrator(config=config).run()

    assert result.adaptive_converged is True
    assert result.truncation_pf_error_bound is not None
    assert result.truncation_pf_error_bound > 1e-3
    assert result.converged is False
    assert result.convergence_reason == "truncation_limited"


def test_auto_bounds_widening_reduces_truncation_and_can_recover_convergence() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(3.0, 0.3),
        s_distribution=ot.Dirac(2.5),
        coarse_points=81,
        u_tail_probability=0.2,
    )
    result = ReliabilityIntegrator(config=config).run()

    initial_abs_bound = abs(float(beta_from_pf(config.u_tail_probability / 2.0, tail="lower")))
    assert result.u_bounds_used is not None
    assert result.u_bounds_used[1] > initial_abs_bound
    assert result.truncation_pf_error_bound is not None
    assert result.converged is True
    assert result.convergence_reason == "converged_relative_tol"


def test_absolute_pf_tolerance_can_recover_adaptive_limited_case() -> None:
    base_kwargs = {
        "r_distribution": ot.Normal(1.0, 1.0),
        "s_distribution": ot.Normal(0.0, 1.0),
        "coarse_points": 41,
        "adaptive_max_depth": 1,
        "adaptive_logpf_tol": 1e-8,
        "u_tail_probability": 1e-12,
    }

    strict_cfg = IntegrationConfig(**base_kwargs)
    strict_result = ReliabilityIntegrator(strict_cfg).run()
    assert strict_result.converged is False
    assert strict_result.convergence_reason == "adaptive_limited"

    relaxed_cfg = IntegrationConfig(**base_kwargs, adaptive_abs_pf_tol=1e-3)
    relaxed_result = ReliabilityIntegrator(relaxed_cfg).run()
    assert relaxed_result.converged is True
    assert relaxed_result.convergence_reason == "converged_absolute_tol"


def test_config_manual_bounds_validation() -> None:
    with pytest.raises(ValueError):
        IntegrationConfig(
            r_distribution=ot.Normal(0.0, 1.0),
            s_distribution=ot.Normal(0.0, 1.0),
            u_manual_bounds=(1.0, 0.0),
        )


def test_config_rejects_legacy_u_min_u_max() -> None:
    with pytest.raises(ValidationError):
        IntegrationConfig(
            r_distribution=ot.Normal(0.0, 1.0),
            s_distribution=ot.Normal(0.0, 1.0),
            u_min=-8.0,
            u_max=8.0,
        )


def test_integrator_runs_with_zero_adaptive_depth() -> None:
    config = _default_config(coarse_points=31, adaptive_max_depth=0)
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    assert math.isfinite(result.beta_star)
    assert result.alpha.shape == (2,)


def test_alpha_is_nan_when_beta_star_is_zero_without_runtime_warning() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        s_distribution=ot.Normal(0.0, 1.0),
        coarse_points=101,
    )
    integrator = ReliabilityIntegrator(config=config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = integrator.run()
    assert all(item.category is not RuntimeWarning for item in caught)
    assert result.beta_star == pytest.approx(0.0)
    assert np.all(np.isnan(result.alpha))
    assert result.design_point_u is None
    assert result.design_point_physical is None


def test_result_includes_physical_design_point_coordinates() -> None:
    config = _default_config(coarse_points=81)
    result = ReliabilityIntegrator(config=config).run()

    assert result.design_point_u is not None
    assert result.design_point_physical is not None

    u1_star = result.design_point_u.u1
    u2_star = result.design_point_u.u2
    assert np.isclose(u1_star, -result.alpha[0] * result.beta_star, rtol=0.0, atol=1e-12)
    assert np.isclose(u2_star, -result.alpha[1] * result.beta_star, rtol=0.0, atol=1e-12)

    physical = result.design_point_physical
    assert np.isfinite(physical.resistance)
    assert np.isfinite(physical.solicitation)
    assert abs(physical.delta_r_minus_s) <= 1e-10

    payload = result.to_dict(include_samples=False)
    assert "design_point_u" in payload
    assert "design_point_physical" in payload
    u_payload = payload["design_point_u"]
    p_payload = payload["design_point_physical"]
    assert isinstance(u_payload, dict)
    assert isinstance(p_payload, dict)
    assert u_payload["u1"] == pytest.approx(u1_star)
    assert u_payload["u2"] == pytest.approx(u2_star)
    assert p_payload["resistance"] == pytest.approx(physical.resistance)
    assert p_payload["solicitation"] == pytest.approx(physical.solicitation)


def test_design_point_solver_optimizes_all_feasible_intervals() -> None:
    config = IntegrationConfig(
        r_distribution=ot.Normal(0.0, 1.0),
        s_distribution=ot.Normal(0.0, 1.0),
        coarse_points=81,
    )
    integrator = ReliabilityIntegrator(config=config)

    # Two feasible basins separated by an infeasible gap.
    # Global optimum is in the right basin at the feasibility boundary.
    def mocked_limit_state_curve(u_values: np.ndarray) -> np.ndarray:
        u = np.asarray(u_values, dtype=float)
        out = np.full_like(u, np.nan, dtype=float)
        left_mask = u <= -1.0
        right_mask = u >= 1.0
        out[left_mask] = (u[left_mask] + 2.0) ** 2 + 0.3
        out[right_mask] = (u[right_mask] - 1.5) ** 2 + 0.2
        return out

    integrator._limit_state_curve = mocked_limit_state_curve  # type: ignore[method-assign]

    beta_star, alpha = integrator._solve_design_point(bounds=(-4.0, 4.0))

    assert math.isfinite(beta_star)

    inferred_u1 = -alpha[0] * beta_star

    # Assert the selected point is in the right feasible basin.
    assert inferred_u1 > 0.0

    # Assert objective quality against the known right-basin boundary optimum
    # (u1=1.0, u2=0.45) while allowing small numerical optimizer variance.
    expected_right_obj = 1.0**2 + 0.45**2
    found_obj = beta_star**2
    assert found_obj <= expected_right_obj + 3e-2

    # Also ensure it beats the left-basin boundary candidate.
    left_boundary_obj = (-1.0) ** 2 + 1.3**2
    assert found_obj < left_boundary_obj


def test_combined_failure_metrics_match_previous_paths() -> None:
    integrator = ReliabilityIntegrator(_default_config(coarse_points=41))
    s_values = np.linspace(-3.0, 4.0, 33)

    combined_f, combined_u1 = integrator._failure_cdf_and_u_for_r_equals_s(s_values, include_u1_equivalent=True)
    assert combined_u1 is not None

    f_ref = integrator._failure_cdf_at_s(s_values)
    u1_ref = integrator._u_for_r_equals_s(s_values)

    assert np.allclose(combined_f, f_ref, rtol=0.0, atol=1e-14)
    assert np.allclose(combined_u1, u1_ref, rtol=0.0, atol=1e-14)


def test_split_interval_batch_matches_manual_interval_formula() -> None:
    config = _default_config(coarse_points=41, adaptive_split_factor=3, adaptive_probe_factor=4)
    integrator = ReliabilityIntegrator(config)
    parent = integrator._build_adaptive_interval(u_left=-1.25, u_right=0.75, prob_weight=0.2, depth=1)

    children = integrator._split_adaptive_interval(parent)

    def _manual_interval(u_left: float, u_right: float, prob_weight: float, depth: int) -> tuple[float, ...]:
        center_u2 = 0.5 * (u_left + u_right)
        center_prob_cdf, center_prob_survival = integrator._normal_probabilities(
            np.array([center_u2], dtype=float),
            compute_cdf=True,
            compute_survival=True,
        )
        center_s = float(
            integrator._map_u_to_distribution(
                np.array([center_u2], dtype=float),
                integrator.s_distribution,
                u_values_cdf=center_prob_cdf,
                u_values_survival=center_prob_survival,
            )[0]
        )
        center_f = float(integrator._failure_cdf_at_s(np.array([center_s], dtype=float))[0])
        center_u1 = float(integrator._u_for_r_equals_s(np.array([center_s], dtype=float))[0])
        center_mass = prob_weight * center_f

        probe_edges = np.linspace(u_left, u_right, config.adaptive_probe_factor + 1)
        probe_cdf, probe_survival = integrator._normal_probabilities(
            probe_edges,
            compute_cdf=True,
            compute_survival=True,
        )
        probe_probs = integrator._u_interval_probabilities(probe_edges, probe_cdf, probe_survival)
        probe_centers = 0.5 * (probe_edges[1:] + probe_edges[:-1])
        probe_center_cdf, probe_center_survival = integrator._normal_probabilities(
            probe_centers,
            compute_cdf=True,
            compute_survival=True,
        )
        probe_s = integrator._map_u_to_distribution(
            probe_centers,
            integrator.s_distribution,
            u_values_cdf=probe_center_cdf,
            u_values_survival=probe_center_survival,
        )
        probe_f = integrator._failure_cdf_at_s(probe_s).reshape(-1)
        probe_mass = float(np.sum(probe_probs.reshape(-1) * probe_f))
        error_est = abs(probe_mass - center_mass)
        refinable = depth < config.adaptive_max_depth and error_est > 0.0
        return (
            float(u_left),
            float(u_right),
            float(prob_weight),
            float(center_u2),
            float(center_s),
            float(center_u1),
            float(probe_mass),
            float(error_est),
            float(refinable),
        )

    split_factor = config.adaptive_split_factor
    edges = np.linspace(parent.u_left, parent.u_right, split_factor + 1)
    edge_cdf, edge_survival = integrator._normal_probabilities(edges, compute_cdf=True, compute_survival=True)
    child_probs = integrator._u_interval_probabilities(edges, edge_cdf, edge_survival)
    manual_children = [
        _manual_interval(float(edges[i]), float(edges[i + 1]), float(child_probs[i]), parent.depth + 1)
        for i in range(split_factor)
        if child_probs[i] > 0.0
    ]

    assert len(children) == len(manual_children)
    for child, manual in zip(children, manual_children):
        assert child.u_left == pytest.approx(manual[0])
        assert child.u_right == pytest.approx(manual[1])
        assert child.prob_weight == pytest.approx(manual[2])
        assert child.center_u2 == pytest.approx(manual[3])
        assert child.center_s == pytest.approx(manual[4])
        assert child.center_u1 == pytest.approx(manual[5])
        assert child.mass_est == pytest.approx(manual[6], rel=0.0, abs=1e-14)
        assert child.error_est == pytest.approx(manual[7], rel=0.0, abs=1e-14)
        assert float(child.refinable) == pytest.approx(manual[8])


def test_integrator_matches_form_reference() -> None:
    rng_state = ot.RandomGenerator.GetState()
    ot.RandomGenerator.SetSeed(FORM_REFERENCE_SEED)
    try:
        config = _default_config(coarse_points=101)
        integrator = ReliabilityIntegrator(config=config)
        near_result = integrator.run()

        beta_form, alpha_form = _form_reference_result(
            config.r_distribution,
            config.s_distribution,
        )
    finally:
        ot.RandomGenerator.SetState(rng_state)

    assert math.isclose(
        near_result.beta_pf,
        beta_form,
        rel_tol=FORM_BETA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )
    assert math.isclose(
        near_result.alpha[0],
        alpha_form[0],
        rel_tol=FORM_ALPHA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )
    assert math.isclose(
        near_result.alpha[1],
        alpha_form[1],
        rel_tol=FORM_ALPHA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )


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


def test_fully_flat_curves_run_end_to_end() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0], [0.5, 0.5, 0.5])
    fragility = FragilityCurve([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=31,
        u_manual_bounds=(-2.0, 2.0),
    )
    result = ReliabilityIntegrator(config=config).run()
    assert np.isfinite(result.pf)
    assert 0.0 <= result.pf <= 1.0


def test_curve_knots_are_injected_into_u_grid_edges() -> None:
    hazard = HazardCurve([0.0, 1.0, 2.0, 3.0], [0.98, 0.8, 0.2, 0.02])
    fragility = FragilityCurve([0.0, 1.0, 2.0, 3.0], [2.0, 0.5, -0.5, -2.0])

    config = IntegrationConfig(
        r_distribution=None,
        s_distribution=None,
        hazard_curve=hazard,
        fragility_curve=fragility,
        coarse_points=5,
        u_manual_bounds=(-6.0, 6.0),
    )
    integrator = ReliabilityIntegrator(config=config)
    grid = integrator._compute_distribution_grid()

    u_bounds = config.u_manual_bounds
    assert u_bounds is not None
    u_min, u_max = u_bounds
    merged_levels = np.unique(np.r_[hazard.hazard_levels.astype(float), fragility.hazard_levels.astype(float)])
    s_cdf = np.array(integrator.s_distribution.computeCDF(merged_levels[:, np.newaxis])).reshape(-1)
    expected_u2 = beta_from_pf(np.clip(s_cdf, 0.0, 1.0), tail="lower")
    expected_u2 = expected_u2[(expected_u2 > u_min) & (expected_u2 < u_max) & np.isfinite(expected_u2)]

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
        u_manual_bounds=(u_min, u_max),
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
        u_manual_bounds=(u_min, u_max),
    )
    integrator = ReliabilityIntegrator(cfg)
    grid = integrator._compute_distribution_grid()

    level_arr = hazard.hazard_levels.astype(float)
    cdf = np.array(integrator.s_distribution.computeCDF(level_arr[:, np.newaxis])).reshape(-1)
    knot_u = beta_from_pf(np.clip(cdf, 0.0, 1.0), tail="lower")
    knot_u = knot_u[np.isfinite(knot_u) & (knot_u > u_min) & (knot_u < u_max)]
    mandatory = integrator._deduplicate_with_min_spacing(  # noqa: SLF001
        np.r_[u_min, u_max, knot_u],
        integrator._MIN_U_EDGE_SPACING,  # noqa: SLF001
    )

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
        u_manual_bounds=(-s_factor, s_factor),
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
    assert math.isclose(
        curve_result.beta_pf,
        dist_result.beta_pf,
        rel_tol=CURVE_VS_DIST_BETA_REL_TOL,
        abs_tol=GENERIC_BETA_ABS_FALLBACK_TOL,
    )
    assert abs(curve_result.beta_pf - analytic_beta) <= abs(dist_result.beta_pf - analytic_beta) + 1e-12


def test_failure_histogram_conserves_probability() -> None:
    config = _default_config(coarse_points=51)
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    water_levels = result.failure_solicitation_levels()
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


def test_result_dict_exposes_convergence_reason() -> None:
    config = _default_config(coarse_points=31)
    result = ReliabilityIntegrator(config=config).run()

    payload = result.to_dict(include_samples=False)
    diagnostics = payload.get("integration_diagnostics")
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("convergence_reason") == result.convergence_reason
