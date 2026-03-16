import numpy as np
import pytest
from failure_paths.common.interp import LinearInterpolator, interpolate_beta_curve
from failure_paths.common.prob import INTERPOLATION_BETA_CAP, pf_from_beta
from failure_paths.reliability.curve_distributions import (
    FragilityDerivedDistribution,
    HazardDerivedDistribution,
)
from failure_paths.reliability.curves import FragilityCurve, HazardCurve


def _example_curves() -> tuple[HazardCurve, FragilityCurve]:
    hazard_levels = [0.0, 1.0, 2.0, 3.0]
    exceedance_probs = [0.99, 0.8, 0.3, 0.02]
    hazard = HazardCurve(hazard_levels, exceedance_probs)

    # Select beta knots directly (monotone)
    beta_knots = [1.3, 0.5, -0.3, -1.5]
    fragility = FragilityCurve(hazard_levels, beta_knots)
    return hazard, fragility


def test_hazard_distribution_matches_curve() -> None:
    hazard, _ = _example_curves()
    distribution = HazardDerivedDistribution(hazard)

    test_levels = np.linspace(hazard.hazard_levels[0], hazard.hazard_levels[-1], 7)
    cdf_vals = np.array(distribution.computeCDF(test_levels[:, np.newaxis])).reshape(test_levels.shape)
    assert np.allclose(cdf_vals, hazard.cdf(test_levels))

    quantile_probs = np.array([0.05, 0.5, 0.95])
    quantiles = np.array(distribution.computeQuantile(quantile_probs)).flatten()
    assert np.allclose(quantiles, hazard.quantile(quantile_probs))


def test_fragility_distribution_matches_curve() -> None:
    _, fragility = _example_curves()
    distribution = FragilityDerivedDistribution(fragility)

    test_levels = np.linspace(fragility.hazard_levels[0], fragility.hazard_levels[-1], 5)
    cdf_vals = np.array(distribution.computeCDF(test_levels[:, np.newaxis])).reshape(test_levels.shape)
    expected = fragility.cdf(test_levels)
    assert np.allclose(cdf_vals, expected)

    quantile_probs = np.array([0.05, 0.4, 0.8])
    quantiles = np.array(distribution.computeQuantile(quantile_probs)).flatten()
    roundtrip = np.array(distribution.computeCDF(quantiles[:, np.newaxis])).reshape(quantiles.shape)
    assert np.allclose(roundtrip, quantile_probs, atol=1e-10)


def test_linear_interpolator_plateau_value() -> None:
    levels = np.array([1.0, 2.0, 3.0, 4.0])
    betas = np.array([4.0, 5.0, 5.0, 6.0])
    interpolator = LinearInterpolator(levels, betas)

    query = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    expected = np.array([4.5, 5.0, 5.0, 5.0, 5.5])
    result = interpolator.value(query)
    assert np.allclose(result, expected)


def test_linear_interpolator_plateau_inverse() -> None:
    beta_knots = np.array([4.0, 5.0, 5.0, 6.0])
    level_knots = np.array([1.0, 2.0, 3.0, 4.0])
    interpolator = LinearInterpolator(beta_knots, level_knots)

    beta_query = np.array([4.5, 5.0, 5.5])
    expected_levels = np.array([1.5, 3.0, 3.5])
    result = interpolator.value(beta_query)
    assert np.allclose(result, expected_levels)


def test_hazard_curve_flat_quantile_is_finite_and_left_endpoint_based() -> None:
    curve = HazardCurve([0.0, 1.0, 2.0], [0.5, 0.5, 0.5])
    quantiles = curve.quantile(np.array([0.4, 0.5, 0.6]))
    assert np.all(np.isfinite(quantiles))
    assert quantiles[1] == pytest.approx(0.0)
    assert quantiles[0] == pytest.approx(0.0)
    assert quantiles[2] == pytest.approx(2.0)


def test_hazard_curve_quantile_reuses_cached_inverse_interpolator() -> None:
    curve = HazardCurve([0.0, 1.0, 2.0, 3.0], [0.95, 0.7, 0.25, 0.05])
    cached = curve._inverse_level_interpolator
    assert cached is not None

    q = np.array([0.1, 0.3, 0.6, 0.9], dtype=float)
    first = curve.quantile(q)
    second = curve.quantile(q)

    assert np.allclose(first, second, rtol=0.0, atol=1e-14)
    assert curve._inverse_level_interpolator is cached


def test_flat_curve_quantile_uses_cached_flat_inverse_path() -> None:
    curve = HazardCurve([0.0, 1.0, 2.0], [0.5, 0.5, 0.5])
    assert curve._inverse_level_interpolator is None
    q = np.array([0.1, 0.5, 0.9], dtype=float)
    out = curve.quantile(q)
    assert np.all(np.isfinite(out))


def test_fragility_curve_requires_non_increasing_betas() -> None:
    with pytest.raises(ValueError):
        FragilityCurve([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])


def test_interpolate_beta_curve_handles_infinite_knots_and_tails() -> None:
    levels = np.array([3.0, 3.000001])
    beta_knots = np.array([np.inf, -np.inf])
    query = np.array([2.999, 3.0, 3.0000005, 3.000001, 3.001])

    beta_values = interpolate_beta_curve(
        levels,
        beta_knots,
        query,
        beta_cap=INTERPOLATION_BETA_CAP,
    )
    pf_values = pf_from_beta(beta_values, tail="upper")

    assert pf_values[0] == pytest.approx(0.0)
    assert pf_values[1] == pytest.approx(0.0)
    assert pf_values[2] == pytest.approx(0.5)
    assert pf_values[3] == pytest.approx(1.0)
    assert pf_values[4] == pytest.approx(1.0)
