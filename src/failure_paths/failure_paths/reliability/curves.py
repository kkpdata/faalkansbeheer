from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import openturns as ot

from ..common.interp import LinearInterpolator


class HazardCurve:
    """Monotone beta-based interpolant mapping hazard levels to cumulative probabilities."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        cumulative_probs: Sequence[float],
        std_normal: ot.Normal | None = None,
    ) -> None:
        if len(hazard_levels) != len(cumulative_probs):
            raise ValueError("hazard_levels and cumulative_probs must have the same length.")
        if len(hazard_levels) < 2:
            raise ValueError("At least two points are required to build a hazard curve.")

        self.std_normal = std_normal or ot.Normal()
        levels = np.asarray(hazard_levels, dtype=float)
        probs = np.asarray(cumulative_probs, dtype=float)
        probs = np.clip(probs, 0, 1)

        if np.any(np.diff(levels) <= 0):
            raise ValueError("hazard_levels must be strictly increasing.")
        if np.any(np.diff(probs) < 0):
            raise ValueError("cumulative_probs must be non-decreasing.")

        beta_knots = self._probabilities_to_beta(probs)

        self.hazard_levels = levels
        self.cdf_values = probs
        self.beta_knots = beta_knots

        self._level_from_beta = LinearInterpolator(beta_knots, levels)
        self._beta_from_level = LinearInterpolator(levels, beta_knots)

    def _probabilities_to_beta(self, probs: np.ndarray | float) -> np.ndarray:
        """Map probabilities onto beta values using the standard normal quantile."""
        probs_arr = np.asarray(probs, dtype=float)
        flat = probs_arr.flatten()
        flat = np.clip(flat, 0, 1)
        betas = np.array(self.std_normal.computeQuantile(flat)).flatten()
        betas = np.nan_to_num(betas)
        return betas.reshape(probs_arr.shape)

    def _beta_to_probabilities(self, beta: np.ndarray | float) -> np.ndarray:
        """Map beta (standard-normal quantiles) back to probabilities."""
        beta_arr = np.asarray(beta, dtype=float)
        flat = beta_arr.flatten()
        probs = np.array(self.std_normal.computeCDF(flat[:, np.newaxis]))
        return probs.reshape(beta_arr.shape)

    def hazard_from_beta(self, beta: np.ndarray | float) -> np.ndarray:
        """Interpolate hazard levels for a given vector of beta values."""
        beta_arr = np.asarray(beta, dtype=float)
        return self._level_from_beta.value(beta_arr)

    def beta_from_hazard(self, hazard: np.ndarray | float) -> np.ndarray:
        """Interpolate beta values for the supplied hazard levels."""
        hazard_arr = np.asarray(hazard, dtype=float)
        return self._beta_from_level.value(hazard_arr)

    def cdf(self, hazard: np.ndarray | float) -> np.ndarray:
        """Evaluate the cumulative distribution at specific hazard levels."""
        hazard_arr = np.asarray(hazard, dtype=float)
        beta_vals = self._beta_from_level.value(hazard_arr)
        return self._beta_to_probabilities(beta_vals)

    def quantile(self, prob: np.ndarray | float) -> np.ndarray:
        """Return hazard levels associated with the supplied exceedance probabilities."""
        prob_arr = np.asarray(prob, dtype=float)
        beta_values = self._probabilities_to_beta(prob_arr)
        return self.hazard_from_beta(beta_values)


class FragilityCurve:
    """Maps beta values to conditional failure probabilities."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        failure_probs: Sequence[float],
        hazard_curve: HazardCurve,
    ) -> None:
        if len(hazard_levels) != len(failure_probs):
            raise ValueError("hazard_levels and failure_probs must have the same length.")
        if len(hazard_levels) < 2:
            raise ValueError("At least two points are required to build a fragility curve.")

        levels = np.asarray(hazard_levels, dtype=float)
        probs = np.asarray(failure_probs, dtype=float)

        if (probs < 0.0).any() or (probs > 1.0).any():
            raise ValueError("failure probabilities must lie within [0, 1].")

        order = np.argsort(levels)
        levels = levels[order]
        probs = probs[order]

        if np.any(np.diff(levels) <= 0):
            raise ValueError("hazard_levels must be strictly increasing for fragility curves.")
        if np.any(np.diff(probs) < 0):
            raise ValueError("failure probabilities must be non-decreasing.")

        beta_knots = hazard_curve.beta_from_hazard(levels)
        self.beta_knots = beta_knots
        self.hazard_levels = levels
        self.failure_probs = probs
        self._failure_from_beta = LinearInterpolator(beta_knots, probs)

    def failure_probability_from_beta(self, beta: np.ndarray | float) -> np.ndarray:
        """Evaluate the conditional failure probability at given beta values."""
        beta_arr = np.asarray(beta, dtype=float)
        return self._failure_from_beta.value(beta_arr)
