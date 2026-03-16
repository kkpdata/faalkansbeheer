import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pytest
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.plotting import plot_integration_diagnostics_1d


def _config(**overrides: object) -> IntegrationConfig:
    kwargs = {
        "r_distribution": ot.Normal(1.0, 1.0),
        "s_distribution": ot.Normal(0.0, 1.0),
        "coarse_points": 41,
    }
    kwargs.update(overrides)
    return IntegrationConfig(**kwargs)


def test_trace_absent_by_default() -> None:
    result = ReliabilityIntegrator(_config()).run()
    assert result.failure_samples.integration_trace is None
    payload = result.to_dict(include_samples=True)
    assert "integration_trace" not in payload


def test_runtime_trace_override_collects_trace() -> None:
    integrator = ReliabilityIntegrator(_config(collect_diagnostics_trace=False))
    result = integrator.run(collect_diagnostics_trace=True)
    trace = result.failure_samples.integration_trace
    assert trace is not None
    assert trace.u_center.size > 0


def test_trace_present_when_enabled_and_mass_is_conserved() -> None:
    result = ReliabilityIntegrator(_config(collect_diagnostics_trace=True)).run()
    trace = result.failure_samples.integration_trace
    assert trace is not None

    n = trace.u_center.size
    assert n > 0
    assert trace.u_left.size == n
    assert trace.u_right.size == n
    assert trace.depth.size == n
    assert trace.prob_weight.size == n
    assert trace.failure_cdf_center.size == n
    assert trace.local_pf_contribution.size == n
    assert trace.error_estimate.size == n
    assert trace.center_u1.size == n

    assert np.all(np.isfinite(trace.u_left))
    assert np.all(np.isfinite(trace.u_right))
    assert np.all(np.isfinite(trace.u_center))
    assert np.all(np.isfinite(trace.failure_cdf_center))
    assert np.all(np.isfinite(trace.local_pf_contribution))
    assert np.all(np.isfinite(trace.error_estimate))
    assert np.all(np.isfinite(trace.center_u1))
    assert np.all(trace.local_pf_contribution >= 0.0)
    assert np.all(trace.error_estimate >= 0.0)
    assert np.all((trace.failure_cdf_center >= 0.0) & (trace.failure_cdf_center <= 1.0))

    total_local_pf = float(np.sum(trace.local_pf_contribution))
    assert np.isclose(total_local_pf, result.pf, rtol=1e-12, atol=1e-15)


def test_result_dict_trace_is_opt_in() -> None:
    result = ReliabilityIntegrator(_config(collect_diagnostics_trace=True)).run()
    trace = result.failure_samples.integration_trace
    assert trace is not None

    payload_default = result.to_dict(include_samples=True)
    assert "integration_trace" not in payload_default

    payload_with_trace = result.to_dict(include_samples=True, include_trace=True)
    assert "integration_trace" in payload_with_trace
    trace_payload = payload_with_trace["integration_trace"]
    assert isinstance(trace_payload, dict)
    assert trace_payload["u_center"]  # non-empty list
    assert len(trace_payload["u_center"]) == trace.u_center.size
    assert len(trace_payload["local_pf_contribution"]) == trace.local_pf_contribution.size


def test_plot_integration_diagnostics_1d_smoke() -> None:
    result = ReliabilityIntegrator(_config(collect_diagnostics_trace=True)).run()
    fig, axes, data = plot_integration_diagnostics_1d(result)
    try:
        assert axes.shape == (2,)
        assert "bin_edges" not in data
        assert "failure_mass" not in data
        assert "conditional_failure" not in data
        assert data["u_center"].size > 0
        cumulative = np.asarray(data["cumulative_pf"], dtype=float)
        assert np.isclose(cumulative[-1], result.pf, rtol=1e-12, atol=1e-15)
    finally:
        plt.close(fig)


def test_plot_integration_diagnostics_1d_requires_trace() -> None:
    result = ReliabilityIntegrator(_config(collect_diagnostics_trace=False)).run()
    with pytest.raises(ValueError, match="collect_diagnostics_trace=True"):
        plot_integration_diagnostics_1d(result)


def test_trace_allows_infinite_center_u1_values() -> None:
    cfg = IntegrationConfig(
        r_distribution=ot.Dirac(0.0),
        s_distribution=ot.Normal(0.0, 1.0),
        coarse_points=101,
        collect_diagnostics_trace=True,
    )
    result = ReliabilityIntegrator(cfg).run()
    trace = result.failure_samples.integration_trace
    assert trace is not None
    assert np.any(np.isinf(trace.center_u1))

    fig, axes, data = plot_integration_diagnostics_1d(result)
    try:
        assert axes.shape == (2,)
        plotted_u1 = np.asarray(data["center_u1"], dtype=float)
        assert np.any(np.isnan(plotted_u1))
    finally:
        plt.close(fig)


def test_trace_does_not_emit_nan_center_u1_for_extreme_tail_case() -> None:
    cfg = IntegrationConfig(
        r_distribution=ot.Gumbel(1.0, 4.0),
        s_distribution=ot.Normal(1.0, 1.0),
        coarse_points=501,
        max_solicitation_level=4.0,
        collect_diagnostics_trace=True,
    )
    result = ReliabilityIntegrator(cfg).run()
    trace = result.failure_samples.integration_trace
    assert trace is not None
    assert not np.any(np.isnan(trace.center_u1))
