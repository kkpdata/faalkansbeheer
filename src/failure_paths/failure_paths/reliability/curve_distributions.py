from __future__ import annotations

import numpy as np
import openturns as ot

from .curves import FragilityCurve, HazardCurve


class HazardDerivedDistribution(ot.PythonDistribution):
    """Treat a HazardCurve as a full OpenTURNS distribution.

    Parameters
    ----------
    hazard_curve : HazardCurve
        Curve describing cumulative probabilities vs hazard levels.
    """

    def __init__(self, hazard_curve: HazardCurve) -> None:
        super().__init__(1)
        self.hazard_curve = hazard_curve

    def computeCDF(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the hazard CDF at one or more levels."""
        return self.hazard_curve.cdf(x)

    def computeSurvivalFunction(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the survival probability ``1 - CDF`` at one or more levels."""
        return self.hazard_curve.survival(x)

    def computeQuantile(self, p: float | np.ndarray, tail: bool = False) -> ot.Point | ot.Sample:
        """Return hazard levels corresponding to the requested probabilities."""
        return self.hazard_curve.quantile(p, tail)

    def getRange(self) -> ot.Interval:
        """Return the hazard range covered by the curve."""
        low = float(self.hazard_curve.hazard_levels[0])
        high = float(self.hazard_curve.hazard_levels[-1])
        return ot.Interval([low], [high])


class FragilityDerivedDistribution(ot.PythonDistribution):
    """Construct an OpenTURNS distribution from a FragilityCurve.

    Parameters
    ----------
    fragility_curve : FragilityCurve
        Curve relating hazard levels to conditional failure probabilities.
    """

    def __init__(self, fragility_curve: FragilityCurve) -> None:
        super().__init__(1)
        self.fragility_curve = fragility_curve

    def computeCDF(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the cumulative failure probability at hazard levels."""
        return self.fragility_curve.cdf(x)

    def computeSurvivalFunction(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the survival probability ``1 - CDF`` at hazard levels."""
        return self.fragility_curve.survival(x)

    def computeQuantile(self, p: float | np.ndarray, tail: bool = False) -> ot.Point | ot.Sample:
        """Return hazard levels whose conditional failure probability equals ``p``."""
        return self.fragility_curve.quantile(p, tail)

    def getRange(self) -> ot.Interval:
        """Return the hazard span covered by the fragility curve."""
        low = float(self.fragility_curve.hazard_levels[0])
        high = float(self.fragility_curve.hazard_levels[-1])
        return ot.Interval([low], [high])
