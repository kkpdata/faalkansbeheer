from __future__ import annotations

import numpy as np
import openturns as ot

from ..common.interp import LinearInterpolator
from .curves import FragilityCurve, HazardCurve


def _as_array(value: object) -> tuple[np.ndarray, bool]:
    """Convert scalars or array-like inputs into flattened arrays.

    Parameters
    ----------
    value : object
        Scalar, list-like, or ndarray input.

    Returns
    -------
    tuple[np.ndarray, bool]
        The flattened ndarray and a flag indicating whether the original input
        was scalar.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1), True
    return arr.reshape(-1), False


def _probabilities(value: object, tail: bool) -> tuple[np.ndarray, bool]:
    """Normalize probability inputs and apply tail handling.

    Parameters
    ----------
    value : object
        Raw probability/survival values (scalar or array-like).
    tail : bool
        ``True`` when the inputs represent upper-tail probabilities.

    Returns
    -------
    tuple[np.ndarray, bool]
        Probabilities clipped to ``[0, 1]`` and the scalar flag.
    """
    arr, was_scalar = _as_array(value)
    if tail:
        arr = 1.0 - arr
    arr = np.clip(arr, 0.0, 1.0)
    return arr, was_scalar


def _quantile_result(values: np.ndarray, was_scalar: bool) -> ot.Point | ot.Sample:
    """Wrap numpy outputs into OpenTURNS containers.

    Parameters
    ----------
    values : np.ndarray
        Quantile results to expose as OpenTURNS values.
    was_scalar : bool
        Whether the originating request targeted a single quantile.

    Returns
    -------
    ot.Point | ot.Sample
        Point for scalar inputs; Sample otherwise.
    """
    if was_scalar:
        return ot.Point([float(values[0])])
    return ot.Sample(values[:, np.newaxis])


class HazardDerivedDistribution(ot.PythonDistribution):
    """Treat a HazardCurve as a full OpenTURNS distribution."""

    def __init__(self, hazard_curve: HazardCurve) -> None:
        """Build a distribution wrapper around a :class:`HazardCurve`.

        Parameters
        ----------
        hazard_curve : HazardCurve
            Curve describing cumulative probabilities vs hazard levels.
        """
        super().__init__(1)
        self.hazard_curve = hazard_curve

    def computeCDF(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the hazard CDF at one or more levels."""
        levels, was_scalar = _as_array(x)
        values = self.hazard_curve.cdf(levels)
        return float(values[0]) if was_scalar else values

    def computeQuantile(self, p: float | np.ndarray, tail: bool = False) -> ot.Point | ot.Sample:
        """Return hazard levels corresponding to the requested probabilities."""
        probs, was_scalar = _probabilities(p, tail)
        levels = self.hazard_curve.quantile(probs)
        return _quantile_result(levels, was_scalar)

    def getRange(self) -> ot.Interval:
        """Return the hazard range covered by the curve."""
        low = float(self.hazard_curve.hazard_levels[0])
        high = float(self.hazard_curve.hazard_levels[-1])
        return ot.Interval([low], [high])


class FragilityDerivedDistribution(ot.PythonDistribution):
    """Construct an OpenTURNS distribution from a FragilityCurve."""

    def __init__(self, fragility_curve: FragilityCurve) -> None:
        """Create a distribution that reuses the fragility beta mappings.

        Parameters
        ----------
        fragility_curve : FragilityCurve
            Curve relating hazard levels to conditional failure probabilities.
        """
        super().__init__(1)
        self.fragility_curve = fragility_curve

        self._level_interp = LinearInterpolator(fragility_curve.hazard_levels, fragility_curve.failure_probs)
        self._beta_interp = LinearInterpolator(fragility_curve.hazard_levels, fragility_curve.beta_knots)
        beta_vals = np.asarray(fragility_curve.beta_knots, dtype=float)
        failure_probs = np.asarray(fragility_curve.failure_probs, dtype=float)
        self._beta_from_failure = LinearInterpolator(failure_probs, beta_vals)

    def computeCDF(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the cumulative failure probability at hazard levels."""
        levels, was_scalar = _as_array(x)
        values = self._level_interp.value(levels)
        return float(values[0]) if was_scalar else values

    def computeQuantile(self, p: float | np.ndarray, tail: bool = False) -> ot.Point | ot.Sample:
        """Return hazard levels whose conditional failure probability equals ``p``."""
        probs, was_scalar = _probabilities(p, tail)
        betas = self._beta_from_failure.value(probs)
        levels = self._beta_interp.inverse(betas)
        return _quantile_result(levels, was_scalar)

    def hazard_level_from_beta(self, beta: float | np.ndarray) -> np.ndarray:
        """Map beta values to hazard levels by inverting the fragility curve."""
        beta_vals, _ = _as_array(beta)
        return self._beta_interp.inverse(beta_vals)

    def beta_from_hazard(self, hazard: float | np.ndarray) -> np.ndarray:
        """Compute beta values for the supplied hazard levels."""
        hazard_vals, _ = _as_array(hazard)
        return self._beta_interp.value(hazard_vals)

    def getRange(self) -> ot.Interval:
        """Return the hazard span covered by the fragility curve."""
        low = float(self.fragility_curve.hazard_levels[0])
        high = float(self.fragility_curve.hazard_levels[-1])
        return ot.Interval([low], [high])
