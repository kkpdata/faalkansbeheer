from __future__ import annotations

from typing import Any

import numpy as np
import openturns as ot
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .curves import FragilityCurve, HazardCurve


class IntegrationConfig(BaseModel):
    """Configuration container for reliability integration runs."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    r_distribution: Any | None = Field(None, description="Distribution of resistance R")
    s_distribution: Any | None = Field(None, description="Distribution of solicitation S")
    max_solicitation_level: float | None = Field(
        default=None,
        description="Optional upper bound on solicitation (water level) used to ignore higher failure mass.",
    )

    u_tail_probability: float = Field(
        1e-23,
        gt=0.0,
        lt=1.0,
        description=(
            "Two-sided tail probability used to derive automatic U-space bounds. "
            "Initial bounds are +/- abs(Phi^-1(u_tail_probability / 2))."
        ),
    )
    u_manual_bounds: tuple[float, float] | None = Field(
        default=None,
        description="Optional explicit U-space bounds (u_min, u_max) overriding automatic bounds.",
    )
    coarse_points: int = Field(101, ge=2, description="Number of U-grid edges (>=2).")
    adaptive_logpf_tol: float = Field(
        1e-4,
        gt=0.0,
        description="Target tolerance for estimated log(Pf) error, i.e. log(1 + delta_pf / pf).",
    )
    adaptive_abs_pf_tol: float | None = Field(
        None,
        ge=0.0,
        description=(
            "Optional absolute Pf error tolerance. "
            "Effective adaptive/global error target is max(relative_target, adaptive_abs_pf_tol)."
        ),
    )
    adaptive_beta_tol: float = Field(
        1e-4,
        gt=0.0,
        description="Stabilization tolerance on consecutive beta_pf estimates.",
    )
    adaptive_max_depth: int = Field(
        8,
        ge=0,
        description="Maximum recursive depth for adaptive mixed-cell refinement.",
    )
    adaptive_split_factor: int = Field(
        4,
        ge=2,
        description="Per-axis split factor applied when refining a mixed cell adaptively.",
    )
    adaptive_probe_factor: int = Field(
        4,
        ge=2,
        description="Per-axis probe resolution used to estimate local mixed-cell error.",
    )
    adaptive_pf_floor: float = Field(
        1e-300,
        gt=0.0,
        description="Numerical floor used when normalizing tiny failure probabilities in adaptive mode.",
    )

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

    @field_validator("u_tail_probability")
    @classmethod
    def _validate_u_tail_probability(cls, value: float) -> float:
        """Ensure the automatic bound tail target is strictly between 0 and 1.

        Parameters
        ----------
        value : float
            Proposed two-sided tail probability.

        Returns
        -------
        float
            Validated probability.

        Raises
        ------
        ValueError
            If ``value`` is not in ``(0, 1)``.
        """
        if not (0.0 < value < 1.0):
            raise ValueError("u_tail_probability must be strictly between 0 and 1.")
        return value

    @field_validator("u_manual_bounds")
    @classmethod
    def _validate_u_manual_bounds(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Validate optional manual U-space bounds."""
        if value is None:
            return value
        lower, upper = float(value[0]), float(value[1])
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError("u_manual_bounds values must be finite.")
        if upper <= lower:
            raise ValueError("u_manual_bounds must be strictly increasing (u_min < u_max).")
        return (lower, upper)

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
