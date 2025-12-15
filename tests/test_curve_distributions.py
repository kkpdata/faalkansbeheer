import numpy as np
from failure_paths.common.interp import LinearInterpolator
from failure_paths.reliability.curve_distributions import (
    FragilityDerivedDistribution,
    HazardDerivedDistribution,
)
from failure_paths.reliability.curves import FragilityCurve, HazardCurve


def _example_curves() -> tuple[HazardCurve, FragilityCurve]:
    hazard_levels = [0.0, 1.0, 2.0, 3.0]
    cumulative_probs = [0.01, 0.2, 0.7, 0.98]
    hazard = HazardCurve(hazard_levels, cumulative_probs)

    fragility_probs = [0.0, 0.1, 0.6, 0.95]
    fragility = FragilityCurve(hazard_levels, fragility_probs, hazard)
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
    hazard, fragility = _example_curves()
    distribution = FragilityDerivedDistribution(fragility)

    test_levels = np.linspace(fragility.hazard_levels[0], fragility.hazard_levels[-1], 5)
    cdf_vals = np.array(distribution.computeCDF(test_levels[:, np.newaxis])).reshape(test_levels.shape)
    expected = np.interp(
        test_levels,
        fragility.hazard_levels,
        fragility.failure_probs,
        left=fragility.failure_probs[0],
        right=fragility.failure_probs[-1],
    )
    assert np.allclose(cdf_vals, expected)

    quantile_probs = np.array([0.05, 0.4, 0.8])
    quantiles = np.array(distribution.computeQuantile(quantile_probs)).flatten()
    roundtrip = np.array(distribution.computeCDF(quantiles[:, np.newaxis])).reshape(quantiles.shape)
    assert np.allclose(roundtrip, quantile_probs, atol=1e-6)


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
