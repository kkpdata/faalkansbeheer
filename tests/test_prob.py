from __future__ import annotations

import numpy as np
import pytest
from failure_paths.common.prob import beta_from_pf, beta_from_pf_stable, clamp_probabilities, pf_from_beta


def test_clamp_probabilities_allows_true_zero_one() -> None:
    probs = np.array([0.0, 0.25, 1.0])
    result = clamp_probabilities(probs)
    assert np.array_equal(result, probs)


def test_beta_from_pf_tail_extremes() -> None:
    probs = np.array([0.0, 0.5, 1.0])

    upper = beta_from_pf(probs, tail="upper")
    assert np.isposinf(upper[0])
    assert np.isfinite(upper[1])
    assert np.isneginf(upper[2])

    lower = beta_from_pf(probs, tail="lower")
    assert np.isneginf(lower[0])
    assert np.isfinite(lower[1])
    assert np.isposinf(lower[2])


def test_pf_from_beta_tail_extremes() -> None:
    beta = np.array([-np.inf, 0.0, np.inf])

    upper = pf_from_beta(beta, tail="upper")
    assert np.array_equal(upper, np.array([1.0, 0.5, 0.0]))

    lower = pf_from_beta(beta, tail="lower")
    assert np.array_equal(lower, np.array([0.0, 0.5, 1.0]))


def test_round_trip_probabilities() -> None:
    probs = np.array([0.0, 1e-12, 0.2, 0.8, 1.0 - 1e-12, 1.0])

    upper = pf_from_beta(beta_from_pf(probs, tail="upper"), tail="upper")
    lower = pf_from_beta(beta_from_pf(probs, tail="lower"), tail="lower")

    exact_mask = (probs == 0.0) | (probs == 1.0)
    approx_mask = ~exact_mask

    assert np.array_equal(upper[exact_mask], probs[exact_mask])
    assert np.array_equal(lower[exact_mask], probs[exact_mask])

    assert np.allclose(upper[approx_mask], probs[approx_mask], rtol=1e-12, atol=0.0)
    assert np.allclose(lower[approx_mask], probs[approx_mask], rtol=1e-12, atol=0.0)


def test_beta_from_pf_stable_preserves_boundary_infinities() -> None:
    near_zero = np.nextafter(0.0, 1.0)
    near_one = np.nextafter(1.0, 0.0)
    probs = np.array([0.0, near_zero, 0.5, near_one, 1.0], dtype=float)

    upper = beta_from_pf_stable(probs, tail="upper", prob_floor=1e-300)
    assert np.isposinf(upper[0])
    assert np.isfinite(upper[1])
    assert np.isfinite(upper[2])
    assert np.isfinite(upper[3])
    assert np.isneginf(upper[4])

    lower = beta_from_pf_stable(probs, tail="lower", prob_floor=1e-300)
    assert np.isneginf(lower[0])
    assert np.isfinite(lower[1])
    assert np.isfinite(lower[2])
    assert np.isfinite(lower[3])
    assert np.isposinf(lower[4])


def test_beta_from_pf_stable_validates_prob_floor() -> None:
    probs = np.array([0.2, 0.8], dtype=float)
    with pytest.raises(ValueError):
        beta_from_pf_stable(probs, prob_floor=0.0)
