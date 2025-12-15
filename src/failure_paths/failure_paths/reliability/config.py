from __future__ import annotations

from typing import Any

import openturns as ot
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .curves import FragilityCurve, HazardCurve


class IntegrationConfig(BaseModel):
    """Configuration container for reliability integration runs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    r_distribution: Any | None = Field(None, description="Distribution of resistance R")
    s_distribution: Any | None = Field(None, description="Distribution of solicitation S")
    threshold: float = Field(0.0, description="Failure threshold for the limit-state function.")

    u_min: float = Field(-8.0, description="Lower bound of the U-space integration range.")
    u_max: float = Field(8.0, description="Upper bound of the U-space integration range.")
    coarse_points: int = Field(65, ge=2, description="Number of U-grid edges (>=2).")
    refine_factor: int = Field(4, ge=1, description="Per-dimension subdivision factor for mixed cells.")

    std_normal: ot.Normal = Field(default_factory=ot.Normal)
    hazard_curve: HazardCurve | None = Field(
        default=None, description="Optional hazard curve overriding R/S distributions."
    )
    fragility_curve: FragilityCurve | None = Field(
        default=None, description="Optional fragility curve operating in beta-space."
    )

    @field_validator("r_distribution", "s_distribution")
    @classmethod
    def _ensure_distribution(cls, value: Any | None) -> ot.Distribution | None:
        """Validate that R and S are OpenTURNS distributions.

        Parameters
        ----------
        value : Any | None
            User supplied distribution (or ``None`` when a curve drives the axis).

        Returns
        -------
        ot.Distribution | None
            Validated OpenTURNS distribution or ``None``.

        Raises
        ------
        TypeError
            If the supplied object does not implement the ``computeCDF`` and ``computeQuantile`` API.
        """
        if value is None:
            return None
        if isinstance(value, ot.Distribution):
            return value
        if hasattr(value, "computeCDF") and hasattr(value, "computeQuantile"):
            return value
        raise TypeError("Expected an OpenTURNS distribution for R and S.")

    @field_validator("u_max")
    @classmethod
    def _validate_range(cls, value: float, info: ValidationInfo) -> float:
        """Ensure the user supplied a valid integration range.

        Parameters
        ----------
        value : float
            Proposed upper bound for the U-grid.
        info : ValidationInfo
            Validation context containing the lower bound.

        Returns
        -------
        float
            Validated ``u_max``.

        Raises
        ------
        ValueError
            If ``value`` is not strictly greater than ``u_min``.
        """
        data = info.data or {}
        u_min = data.get("u_min")
        if u_min is not None and value <= u_min:
            raise ValueError("u_max must be greater than u_min.")
        return value

    @model_validator(mode="after")
    def _ensure_axis_sources(self) -> IntegrationConfig:
        """Ensure each axis is driven by exactly one source (curve or distribution)."""
        r_has_dist = self.r_distribution is not None
        r_has_curve = self.fragility_curve is not None
        if r_has_dist and r_has_curve:
            raise ValueError("Provide either r_distribution or fragility_curve for resistance, not both.")
        if not r_has_dist and not r_has_curve:
            raise ValueError("r_distribution or fragility_curve is required for resistance.")

        s_has_dist = self.s_distribution is not None
        s_has_curve = self.hazard_curve is not None
        if s_has_dist and s_has_curve:
            raise ValueError("Provide either s_distribution or hazard_curve for solicitation, not both.")
        if not s_has_dist and not s_has_curve:
            raise ValueError("s_distribution or hazard_curve is required for solicitation.")

        return self
