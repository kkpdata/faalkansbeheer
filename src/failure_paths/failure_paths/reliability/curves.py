from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import openturns as ot

from ..common.interp import LinearInterpolator


class _BetaCurveBase:
    """Shared helper for hazard/fragility curves storing beta/level interpolators."""

    def __init__(self, hazard_levels: Sequence[float], betas: Sequence[float]) -> None:
        if len(hazard_levels) < 2:
            raise ValueError("At least two points are required to build a curve.")
        if len(hazard_levels) != len(betas):
            raise ValueError("hazard_levels and betas must have the same length.")
        if np.any(np.diff(hazard_levels) <= 0):
            raise ValueError("hazard_levels must be strictly increasing.")

        self.std_normal = ot.Normal()
        self.hazard_levels = np.asarray(hazard_levels, dtype=float)
        self.beta_knots = np.asarray(betas, dtype=float)
        self._level_from_beta = LinearInterpolator(self.beta_knots, self.hazard_levels)
        self._beta_from_level = LinearInterpolator(self.hazard_levels, self.beta_knots)

    def probabilities_to_beta(self, probs: np.ndarray | float, tail: bool = False) -> np.ndarray:
        probs_arr = np.asarray(probs, dtype=float)
        flat = np.clip(probs_arr.flatten(), 0, 1)
        betas = np.array(self.std_normal.computeQuantile(flat)).flatten()
        betas = np.nan_to_num(betas)
        if tail:
            betas = -1 * betas
        return betas.reshape(probs_arr.shape)

    def beta_to_probabilities(self, beta: np.ndarray | float) -> np.ndarray:
        beta_arr = np.asarray(beta, dtype=float)
        flat = beta_arr.flatten()
        probs = np.array(self.std_normal.computeCDF(flat[:, np.newaxis]))
        return probs.reshape(beta_arr.shape)

    def cdf(self, hazard: np.ndarray | float) -> np.ndarray:
        hazard_arr = np.asarray(hazard, dtype=float)
        beta_vals = self._beta_from_level.value(hazard_arr)
        flat = beta_vals.flatten()
        probs = np.array(self.std_normal.computeCDF(flat[:, np.newaxis]))
        return probs.reshape(hazard_arr.shape)

    def quantile(self, prob: np.ndarray | float, tail: bool = False) -> np.ndarray:
        prob_arr = np.asarray(prob, dtype=float)
        beta_values = self.probabilities_to_beta(prob_arr, tail=tail)
        return self._level_from_beta.value(beta_values)

    def hazard_from_beta(self, beta: np.ndarray | float) -> np.ndarray:
        """Expose level interpolation publicly."""
        beta_arr = np.asarray(beta, dtype=float)
        return self._level_from_beta.value(beta_arr)

    def beta_from_hazard(self, hazard: np.ndarray | float) -> np.ndarray:
        """Expose beta interpolation publicly."""
        hazard_arr = np.asarray(hazard, dtype=float)
        return self._beta_from_level.value(hazard_arr)


class HazardCurve(_BetaCurveBase):
    """Maps hazard levels to exceedance probabilities (cumulative distribution)."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        exceedance_probs: Sequence[float],
    ) -> None:
        probs = np.asarray(exceedance_probs, dtype=float)
        probs = np.clip(probs, 0, 1)
        if np.any(np.diff(probs) > 0):
            raise ValueError("exceedance_probs must be non-increasing.")
        betas = np.array(ot.Normal().computeQuantile(probs, True)).flatten()
        super().__init__(hazard_levels, betas)


class FragilityCurve(_BetaCurveBase):
    """Maps hazard levels to reliability indices assuming Pf = Φ(-β)."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        betas: Sequence[float],
    ) -> None:
        super().__init__(hazard_levels, betas)
