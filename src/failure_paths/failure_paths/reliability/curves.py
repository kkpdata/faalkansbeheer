from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import openturns as ot

from ..common.interp import LinearInterpolator
from ..common.prob import beta_from_pf, pf_from_beta


def _left_endpoint_inverse_monotone(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_query: np.ndarray | float,
) -> np.ndarray | float:
    """Evaluate a monotone inverse using left-endpoint generalized-inverse semantics.

    Parameters
    ----------
    x_nodes : np.ndarray
        Monotone non-decreasing x knots.
    y_nodes : np.ndarray
        Associated y knots.
    x_query : np.ndarray | float
        Query values in x-space.

    Returns
    -------
    np.ndarray | float
        Interpolated y values. Plateaus in ``x_nodes`` map to the left endpoint.

    Raises
    ------
    ValueError
        If knot arrays have different lengths, are empty, or ``x_nodes`` is not
        monotone non-decreasing.
    """
    x_arr = np.asarray(x_nodes, dtype=float).reshape(-1)
    y_arr = np.asarray(y_nodes, dtype=float).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x_nodes and y_nodes must have identical lengths.")
    if x_arr.size == 0:
        raise ValueError("At least one knot is required.")
    if np.any(np.diff(x_arr) < 0):
        raise ValueError("x_nodes must be non-decreasing.")

    is_scalar = np.isscalar(x_query)
    q_arr = np.asarray(x_query, dtype=float)
    q_flat = q_arr.reshape(-1)

    unique_x, first_idx = np.unique(x_arr, return_index=True)
    unique_y = y_arr[first_idx]

    if unique_x.size == 1:
        # Fully flat inverse relation: keep finite outputs while preserving
        # left-endpoint semantics at/below the plateau value.
        out_flat = np.where(q_flat <= unique_x[0], unique_y[0], y_arr[-1]).astype(float)
    else:
        out_flat = np.asarray(LinearInterpolator(unique_x, unique_y).value(q_flat), dtype=float).reshape(-1)

    out = out_flat.reshape(q_arr.shape)
    if is_scalar:
        return float(out.reshape(-1)[0])
    return out


class _BetaCurveBase(ABC):
    """Shared helper for hazard/fragility curves storing beta/level interpolators."""

    def __init__(self, beta_inf_cap: float = 1e6) -> None:
        self.std_normal = ot.Normal()
        self.beta_inf_cap = float(beta_inf_cap)
        self._inverse_beta_knots = np.array([], dtype=float)
        self._inverse_level_knots = np.array([], dtype=float)
        self._inverse_unique_beta_knots = np.array([], dtype=float)
        self._inverse_unique_level_knots = np.array([], dtype=float)
        self._inverse_level_interpolator: LinearInterpolator | None = None
        self._inverse_flat_beta: float | None = None
        self._inverse_flat_left_level: float | None = None
        self._inverse_flat_right_level: float | None = None

    def _sanitize_beta_knots(self, betas: np.ndarray) -> np.ndarray:
        return np.nan_to_num(betas, posinf=self.beta_inf_cap, neginf=-self.beta_inf_cap)

    @abstractmethod
    def set_levels(self, hazard_levels: Sequence[float], betas: Sequence[float]):
        pass

    def _validate_levels(self, hazard_levels: Sequence[float], betas: Sequence[float]):
        if len(hazard_levels) < 2:
            raise ValueError("At least two points are required to build a curve.")
        if len(hazard_levels) != len(betas):
            raise ValueError("hazard_levels and betas must have the same length.")
        if np.any(np.diff(hazard_levels) <= 0):
            raise ValueError("hazard_levels must be strictly increasing.")

    def probabilities_to_beta(self, probs: np.ndarray | float, tail: bool = False) -> np.ndarray:
        probs_arr = np.asarray(probs, dtype=float)
        probs_arr = np.clip(probs_arr, 0, 1)
        if tail:
            betas = beta_from_pf(probs_arr, tail="upper")
        else:
            betas = beta_from_pf(probs_arr, tail="lower")
        return np.nan_to_num(betas, nan=0.0, posinf=np.inf, neginf=-np.inf)

    def beta_to_probabilities(self, beta: np.ndarray | float) -> np.ndarray:
        beta_arr = np.asarray(beta, dtype=float)
        return pf_from_beta(beta_arr, tail="lower")

    def cdf(self, hazard: np.ndarray | float) -> np.ndarray:
        hazard_arr = np.asarray(hazard, dtype=float)
        beta_vals = self._beta_from_level.value(hazard_arr)
        return pf_from_beta(beta_vals, tail="lower")

    def survival(self, hazard: np.ndarray | float) -> np.ndarray:
        hazard_arr = np.asarray(hazard, dtype=float)
        beta_vals = self._beta_from_level.value(hazard_arr)
        return pf_from_beta(beta_vals, tail="upper")

    def quantile(self, prob: np.ndarray | float, tail: bool = False) -> np.ndarray:
        prob_arr = np.asarray(prob, dtype=float)
        beta_values = self.probabilities_to_beta(prob_arr, tail=tail)
        return self._inverse_levels_from_beta(beta_values)

    def hazard_from_beta(self, beta: np.ndarray | float) -> np.ndarray:
        """Expose level interpolation publicly."""
        beta_arr = np.asarray(beta, dtype=float)
        return self._level_from_beta.value(beta_arr)

    def beta_from_hazard(self, hazard: np.ndarray | float) -> np.ndarray:
        """Expose beta interpolation publicly."""
        hazard_arr = np.asarray(hazard, dtype=float)
        return self._beta_from_level.value(hazard_arr)

    def _set_inverse_mapping(self, beta_knots: np.ndarray, level_knots: np.ndarray) -> None:
        beta_arr = np.asarray(beta_knots, dtype=float).reshape(-1)
        level_arr = np.asarray(level_knots, dtype=float).reshape(-1)
        if beta_arr.size != level_arr.size:
            raise ValueError("Inverse mapping knots must have identical lengths.")

        self._inverse_beta_knots = beta_arr
        self._inverse_level_knots = level_arr

        unique_beta, first_idx = np.unique(beta_arr, return_index=True)
        unique_level = level_arr[first_idx]
        self._inverse_unique_beta_knots = unique_beta
        self._inverse_unique_level_knots = unique_level

        if unique_beta.size == 1:
            self._inverse_level_interpolator = None
            self._inverse_flat_beta = float(unique_beta[0])
            self._inverse_flat_left_level = float(unique_level[0])
            self._inverse_flat_right_level = float(level_arr[-1])
        else:
            self._inverse_level_interpolator = LinearInterpolator(unique_beta, unique_level)
            self._inverse_flat_beta = None
            self._inverse_flat_left_level = None
            self._inverse_flat_right_level = None

    def _inverse_levels_from_beta(self, beta_values: np.ndarray | float) -> np.ndarray:
        beta_arr = np.asarray(beta_values, dtype=float)
        is_scalar = np.isscalar(beta_values)
        beta_flat = beta_arr.reshape(-1)

        if self._inverse_unique_beta_knots.size == 0:
            raise RuntimeError("Inverse mapping is not initialized.")

        if self._inverse_level_interpolator is None:
            if (
                self._inverse_flat_beta is None
                or self._inverse_flat_left_level is None
                or self._inverse_flat_right_level is None
            ):
                raise RuntimeError("Flat inverse mapping cache is not initialized.")
            level_flat = np.where(
                beta_flat <= self._inverse_flat_beta,
                self._inverse_flat_left_level,
                self._inverse_flat_right_level,
            ).astype(float)
        else:
            level_flat = np.asarray(self._inverse_level_interpolator.value(beta_flat), dtype=float).reshape(-1)

        out = level_flat.reshape(beta_arr.shape)
        if is_scalar:
            return float(out.reshape(-1)[0])
        return out


class HazardCurve(_BetaCurveBase):
    """Maps hazard levels to exceedance probabilities (cumulative distribution)."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        exceedance_probs: Sequence[float],
        *,
        beta_inf_cap: float = 1e6,
    ) -> None:
        super().__init__(beta_inf_cap=beta_inf_cap)
        self.set_levels(hazard_levels, exceedance_probs)

    def set_levels(self, hazard_levels: Sequence[float], exceedance_probs: Sequence[float]):
        probs = np.asarray(exceedance_probs, dtype=float)
        probs = np.clip(probs, 0, 1)
        if np.any(np.diff(probs) > 0):
            raise ValueError("exceedance_probs must be non-increasing.")
        betas = beta_from_pf(probs, tail="upper")

        betas = self._sanitize_beta_knots(betas)
        self._validate_levels(hazard_levels, betas)
        self.hazard_levels = np.asarray(hazard_levels, dtype=float)
        self.beta_knots = np.asarray(betas, dtype=float)
        self._level_from_beta = LinearInterpolator(self.beta_knots, self.hazard_levels)
        self._beta_from_level = LinearInterpolator(self.hazard_levels, self.beta_knots)
        self._set_inverse_mapping(self.beta_knots, self.hazard_levels)


class FragilityCurve(_BetaCurveBase):
    """Maps hazard levels to reliability indices assuming Pf = Φ(-β)."""

    def __init__(
        self,
        hazard_levels: Sequence[float],
        betas: Sequence[float],
        *,
        beta_inf_cap: float = 1e6,
    ) -> None:
        super().__init__(beta_inf_cap=beta_inf_cap)
        self.set_levels(hazard_levels, betas)

    def set_levels(self, hazard_levels: Sequence[float], betas: Sequence[float]):
        betas = self._sanitize_beta_knots(np.asarray(betas, dtype=float))
        if np.any(np.diff(betas) > 0):
            raise ValueError("betas must be non-increasing.")
        self._validate_levels(hazard_levels, betas)
        self.hazard_levels = np.asarray(hazard_levels, dtype=float)
        self.beta_knots = np.asarray(betas, dtype=float)
        self._level_from_beta = LinearInterpolator(self.beta_knots[::-1], self.hazard_levels[::-1])
        self._beta_from_level = LinearInterpolator(self.hazard_levels, self.beta_knots)
        self._set_inverse_mapping(self.beta_knots[::-1], self.hazard_levels[::-1])

    def probabilities_to_beta(self, probs: np.ndarray | float, tail: bool = False) -> np.ndarray:
        """Interpret probabilities as Pf (lower tail) or 1-Pf (upper tail)."""
        prob_arr = np.asarray(probs, dtype=float)
        base = super().probabilities_to_beta(prob_arr, tail=False)
        if tail:
            return base
        return -base.reshape(prob_arr.shape)

    def beta_to_probabilities(self, beta: np.ndarray | float) -> np.ndarray:
        """Return conditional failure probabilities Pf = Φ(-β)."""
        beta_arr = np.asarray(beta, dtype=float)
        return super().beta_to_probabilities(-beta_arr)

    def cdf(self, hazard: np.ndarray | float) -> np.ndarray:
        """CDF equals Pf = Φ(-β)."""
        return super().survival(hazard)

    def survival(self, hazard: np.ndarray | float) -> np.ndarray:
        """Survival equals 1 - Pf = Φ(β)."""
        return super().cdf(hazard)
