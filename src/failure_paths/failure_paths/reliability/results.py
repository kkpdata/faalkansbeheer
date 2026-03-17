from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DesignPointUSpace:
    """Design point coordinates in transformed standard-normal space.

    Attributes
    ----------
    u1 : float
        Resistance coordinate ``u1 = Phi^{-1}(F_R(r))``.
    u2 : float
        Solicitation coordinate ``u2 = Phi^{-1}(F_S(s))``.
    """

    u1: float
    u2: float

    def to_dict(self) -> dict[str, float]:
        """Serialize U-space design point coordinates.

        Returns
        -------
        dict[str, float]
            JSON-serializable mapping with ``u1`` and ``u2`` values.
        """
        return {"u1": float(self.u1), "u2": float(self.u2)}


@dataclass
class DesignPointPhysical:
    """Design point coordinates in physical resistance/solicitation space.

    Attributes
    ----------
    resistance : float
        Physical resistance value at the design point.
    solicitation : float
        Physical solicitation value at the design point.
    delta_r_minus_s : float
        Difference ``resistance - solicitation``; should be near zero on the
        limit-state boundary.
    """

    resistance: float
    solicitation: float
    delta_r_minus_s: float

    def to_dict(self) -> dict[str, float]:
        """Serialize physical design point coordinates.

        Returns
        -------
        dict[str, float]
            JSON-serializable mapping of physical design-point values.
        """
        return {
            "resistance": float(self.resistance),
            "solicitation": float(self.solicitation),
            "delta_r_minus_s": float(self.delta_r_minus_s),
        }


@dataclass
class IntegrationDiagnosticsTrace:
    """Interval-level diagnostics trace from the 1D adaptive integrator.

    Notes
    -----
    Coordinate convention used throughout this module:

    - ``u1 = Phi^{-1}(F_R(r))`` is the resistance standard-normal coordinate.
    - ``u2 = Phi^{-1}(F_S(s))`` is the solicitation standard-normal coordinate.

    Attributes
    ----------
    u_left : np.ndarray
        Left edge of each final integration interval in standard-normal ``u2`` space.
    u_right : np.ndarray
        Right edge of each final integration interval in standard-normal ``u2`` space.
    u_center : np.ndarray
        Center point of each interval in standard-normal ``u2`` space.
    depth : np.ndarray
        Refinement depth per interval. Larger values indicate more splitting effort.
    prob_weight : np.ndarray
        Probability mass of each ``u2`` interval under the solicitation distribution.
    failure_cdf_center : np.ndarray
        Failure indicator signal at interval center, represented as ``F_R(s_center)``
        and bounded in ``[0, 1]``.
    local_pf_contribution : np.ndarray
        Local failure probability contribution from each interval. The sum over all
        intervals matches integrated ``Pf`` within numerical tolerance.
    error_estimate : np.ndarray
        Local adaptive error estimate assigned to each interval.
    center_u1 : np.ndarray
        Equivalent ``u1`` location on the limit-state boundary (``R = S``) evaluated at
        ``u_center``. Values can be finite or ``+/-inf`` in extreme tails.
    """

    u_left: np.ndarray
    u_right: np.ndarray
    u_center: np.ndarray
    depth: np.ndarray
    prob_weight: np.ndarray
    failure_cdf_center: np.ndarray
    local_pf_contribution: np.ndarray
    error_estimate: np.ndarray
    center_u1: np.ndarray

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        """Serialize diagnostics trace arrays to plain Python lists.

        Returns
        -------
        dict[str, list[float] | list[int]]
            JSON-serializable mapping of trace arrays.
        """
        return {
            "u_left": self.u_left.tolist(),
            "u_right": self.u_right.tolist(),
            "u_center": self.u_center.tolist(),
            "depth": self.depth.tolist(),
            "prob_weight": self.prob_weight.tolist(),
            "failure_cdf_center": self.failure_cdf_center.tolist(),
            "local_pf_contribution": self.local_pf_contribution.tolist(),
            "error_estimate": self.error_estimate.tolist(),
            "center_u1": self.center_u1.tolist(),
        }

    def to_user_dict(self, include_legacy_names: bool = False) -> dict[str, list[float] | list[int] | dict[str, str]]:
        """Serialize diagnostics trace with user-friendly key names.

        Parameters
        ----------
        include_legacy_names : bool
            Whether to include the legacy compact key names alongside the
            user-friendly schema.

        Returns
        -------
        dict[str, list[float] | list[int] | dict[str, str]]
            JSON-serializable mapping with descriptive trace keys.
        """
        user_data: dict[str, list[float] | list[int] | dict[str, str]] = {
            "interval_left_u2": self.u_left.tolist(),
            "interval_right_u2": self.u_right.tolist(),
            "interval_center_u2": self.u_center.tolist(),
            "refinement_depth": self.depth.tolist(),
            "interval_probability_mass": self.prob_weight.tolist(),
            "failure_probability_at_center": self.failure_cdf_center.tolist(),
            "local_failure_probability": self.local_pf_contribution.tolist(),
            "local_error_estimate": self.error_estimate.tolist(),
            "equivalent_u1_on_limit_state": self.center_u1.tolist(),
            "units": {
                "interval_left_u2": "standard normal solicitation coordinate u2 = Phi^{-1}(F_S(s))",
                "interval_right_u2": "standard normal solicitation coordinate u2 = Phi^{-1}(F_S(s))",
                "interval_center_u2": "standard normal solicitation coordinate u2 = Phi^{-1}(F_S(s))",
                "refinement_depth": "interval split depth (integer)",
                "interval_probability_mass": "probability mass",
                "failure_probability_at_center": "conditional failure probability",
                "local_failure_probability": "failure probability contribution",
                "local_error_estimate": "failure probability contribution",
                "equivalent_u1_on_limit_state": "standard normal resistance coordinate u1 = Phi^{-1}(F_R(r))",
            },
        }
        if include_legacy_names:
            user_data["legacy"] = self.to_dict()
        return user_data


@dataclass
class FailureSamples:
    """Weighted failure samples plus integration diagnostics.

    Attributes
    ----------
    weights : np.ndarray
        Failure probability weights. The sum equals integrated ``Pf``.
    points : np.ndarray
        Representative failure points in standard-normal space with shape ``(N, 2)``.
        Column 0 is ``u1`` (resistance coordinate, ``Phi^{-1}(F_R(r))``) and
        column 1 is ``u2`` (solicitation coordinate, ``Phi^{-1}(F_S(s))``).
    coarse_fail_cells : int
        Number of intervals identified as failing at coarse resolution.
    refined_fail_cells : int
        Number of failing intervals after adaptive refinement.
    mixed_cells : int
        Number of intervals that required mixed/fractional failure treatment.
    hazard_levels : np.ndarray | None
        Physical hazard levels mapped from failure sample points.
    solicitation_levels : np.ndarray | None
        Physical solicitation levels mapped from failure sample points.
    adaptive_converged : bool | None
        Whether adaptive interval refinement converged for the solved bounds pass.
    adaptive_iterations : int
        Number of adaptive refinement iterations.
    adaptive_estimated_logpf_error : float | None
        Adaptive relative-error proxy in log space ``log(1 + err/Pf)``.
    adaptive_remaining_pf_error : float | None
        Remaining absolute adaptive error bound on ``Pf``.
    adaptive_max_depth_reached_cells : int
        Count of intervals that reached maximum adaptive refinement depth.
    converged : bool | None
        Global convergence status including truncation and adaptive criteria.
    convergence_reason : str | None
        Text reason for global convergence status.
    estimated_logpf_error : float | None
        Combined relative-error proxy in log space ``log(1 + err/Pf)``.
    truncation_pf_error_bound : float | None
        Absolute ``Pf`` error bound from finite ``u``-domain truncation.
    u_bounds_used : tuple[float, float] | None
        Final standard-normal integration bounds ``(u_min, u_max)``.
    integration_trace : IntegrationDiagnosticsTrace | None
        Optional per-interval diagnostics trace (opt-in collection).
    """

    weights: np.ndarray
    points: np.ndarray
    coarse_fail_cells: int = 0
    refined_fail_cells: int = 0
    mixed_cells: int = 0
    hazard_levels: np.ndarray | None = None
    solicitation_levels: np.ndarray | None = None
    adaptive_converged: bool | None = None
    adaptive_iterations: int = 0
    adaptive_estimated_logpf_error: float | None = None
    adaptive_remaining_pf_error: float | None = None
    adaptive_max_depth_reached_cells: int = 0
    converged: bool | None = None
    convergence_reason: str | None = None
    estimated_logpf_error: float | None = None
    truncation_pf_error_bound: float | None = None
    u_bounds_used: tuple[float, float] | None = None
    integration_trace: IntegrationDiagnosticsTrace | None = None

    @property
    def coarse_fail_intervals(self) -> int:
        """Return coarse failing interval count (alias for ``coarse_fail_cells``)."""
        return self.coarse_fail_cells

    @property
    def refined_fail_intervals(self) -> int:
        """Return refined failing interval count (alias for ``refined_fail_cells``)."""
        return self.refined_fail_cells

    @property
    def transition_intervals(self) -> int:
        """Return mixed/transition interval count (alias for ``mixed_cells``)."""
        return self.mixed_cells

    @property
    def adaptive_max_depth_reached_intervals(self) -> int:
        """Return max-depth interval count (alias for ``adaptive_max_depth_reached_cells``)."""
        return self.adaptive_max_depth_reached_cells

    def __repr__(self) -> str:
        """Summarize weights and diagnostics."""
        adaptive_bits = ""
        if self.adaptive_converged is not None:
            adaptive_bits = (
                f", adaptive_converged={self.adaptive_converged}, "
                f"adaptive_iterations={self.adaptive_iterations}, "
                f"adaptive_max_depth_cells={self.adaptive_max_depth_reached_cells}"
            )
        global_bits = ""
        if self.converged is not None:
            global_err = f"{self.estimated_logpf_error:.3e}" if self.estimated_logpf_error is not None else "None"
            global_bits = f", converged={self.converged}, global_err={global_err}"
            if self.convergence_reason is not None:
                global_bits += f", reason={self.convergence_reason}"
        trace_bits = ""
        if self.integration_trace is not None:
            trace_bits = f", trace_intervals={self.integration_trace.u_center.size}"
        return (
            "FailureSamples("
            f"count={self.weights.size}, "
            f"coarse_fail_cells={self.coarse_fail_cells}, "
            f"refined_fail_cells={self.refined_fail_cells}, "
            f"mixed_cells={self.mixed_cells}"
            f"{adaptive_bits}"
            f"{global_bits}"
            f"{trace_bits}"
            ")"
        )


@dataclass
class IntegrationResult:
    """Reliability metrics and diagnostics produced by the 1D adaptive integrator.

    Notes
    -----
    Coordinate convention:

    - ``u1 = Phi^{-1}(F_R(r))`` (resistance axis).
    - ``u2 = Phi^{-1}(F_S(s))`` (solicitation axis).

    Attributes
    ----------
    pf : float
        Estimated failure probability.
    beta_star : float
        Reliability index at the design point.
    alpha : np.ndarray
        Design-point direction cosines in standard-normal space.
    failure_samples : FailureSamples
        Weighted failure sample representation and diagnostics.
    beta_pf : float
        Reliability index mapped directly from ``pf``.
    hazard_level : float | None
        Hazard level corresponding to ``pf`` when available from input model.
    adaptive_converged : bool | None
        Whether adaptive refinement converged in the final bounds pass.
    adaptive_iterations : int
        Number of adaptive refinement iterations.
    adaptive_estimated_logpf_error : float | None
        Adaptive relative-error proxy in log space ``log(1 + err/Pf)``.
    adaptive_remaining_pf_error : float | None
        Remaining absolute adaptive error bound on ``Pf``.
    adaptive_max_depth_reached_cells : int
        Count of intervals that reached maximum adaptive refinement depth.
    converged : bool | None
        Global convergence status including truncation and adaptive criteria.
    convergence_reason : str | None
        Text reason for global convergence status.
    estimated_logpf_error : float | None
        Combined relative-error proxy in log space ``log(1 + err/Pf)``.
    truncation_pf_error_bound : float | None
        Absolute ``Pf`` error bound from finite ``u``-domain truncation.
    u_bounds_used : tuple[float, float] | None
        Final standard-normal integration bounds ``(u_min, u_max)``.
    design_point_u : DesignPointUSpace | None
        Design point coordinates in U-space, or ``None`` when undefined.
    design_point_physical : DesignPointPhysical | None
        Design point coordinates in physical space, or ``None`` when undefined.
    """

    pf: float
    beta_star: float
    alpha: np.ndarray
    failure_samples: FailureSamples
    beta_pf: float
    hazard_level: float | None = None
    adaptive_converged: bool | None = None
    adaptive_iterations: int = 0
    adaptive_estimated_logpf_error: float | None = None
    adaptive_remaining_pf_error: float | None = None
    adaptive_max_depth_reached_cells: int = 0
    converged: bool | None = None
    convergence_reason: str | None = None
    estimated_logpf_error: float | None = None
    truncation_pf_error_bound: float | None = None
    u_bounds_used: tuple[float, float] | None = None
    design_point_u: DesignPointUSpace | None = None
    design_point_physical: DesignPointPhysical | None = None

    @property
    def u_integration_bounds(self) -> tuple[float, float] | None:
        """Return integration bounds alias for ``u_bounds_used``."""
        return self.u_bounds_used

    @property
    def truncation_error_bound_pf(self) -> float | None:
        """Return truncation error alias for ``truncation_pf_error_bound``."""
        return self.truncation_pf_error_bound

    def summary(self) -> dict[str, float | str | list[float] | None]:
        """Return headline reliability metrics with descriptive key names."""
        data: dict[str, float | str | list[float] | None] = {
            "failure_probability": float(self.pf),
            "design_point_reliability_index": float(self.beta_star),
            "pf_reliability_index": float(self.beta_pf),
            "alpha": self.alpha.tolist(),
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "total_error_log1p_relative": (
                float(self.estimated_logpf_error) if self.estimated_logpf_error is not None else None
            ),
            "truncation_pf_error_bound": (
                float(self.truncation_pf_error_bound) if self.truncation_pf_error_bound is not None else None
            ),
            "adaptive_converged": self.adaptive_converged,
            "adaptive_error_log1p_relative": (
                float(self.adaptive_estimated_logpf_error) if self.adaptive_estimated_logpf_error is not None else None
            ),
        }
        if self.design_point_u is not None:
            data["design_point_u1"] = float(self.design_point_u.u1)
            data["design_point_u2"] = float(self.design_point_u.u2)
        if self.design_point_physical is not None:
            data["design_point_resistance"] = float(self.design_point_physical.resistance)
            data["design_point_solicitation"] = float(self.design_point_physical.solicitation)
        return data

    def to_dict(self, include_samples: bool = False, include_trace: bool = False) -> dict[str, object]:
        """Serialize integration results with user-friendly key names.

        Parameters
        ----------
        include_samples : bool
            Whether to include the failure samples and weights.
        include_trace : bool
            Whether to include interval-level integration diagnostics trace.

        Returns
        -------
        dict[str, object]
            JSON-serializable structure with descriptive naming.
        """
        return self.to_user_dict(
            include_samples=include_samples,
            include_trace=include_trace,
            include_legacy_names=False,
        )

    def to_user_dict(
        self,
        include_samples: bool = False,
        include_trace: bool = False,
        include_legacy_names: bool = False,
    ) -> dict[str, object]:
        """Serialize the integration result with descriptive user-facing keys.

        Parameters
        ----------
        include_samples : bool
            Whether to include failure sample arrays.
        include_trace : bool
            Whether to include interval-level diagnostics trace.
        include_legacy_names : bool
            Whether to include the legacy compact payload under ``legacy`` for
            backward compatibility during migration.

        Returns
        -------
        dict[str, object]
            JSON-serializable payload with descriptive naming and optional legacy
            compatibility data.
        """
        diagnostics = {
            "coarse_fail_intervals": self.failure_samples.coarse_fail_cells,
            "refined_fail_intervals": self.failure_samples.refined_fail_cells,
            "transition_intervals": self.failure_samples.mixed_cells,
            "adaptive_converged": self.adaptive_converged,
            "adaptive_iterations": self.adaptive_iterations,
            "adaptive_error_log1p_relative": (
                float(self.adaptive_estimated_logpf_error) if self.adaptive_estimated_logpf_error is not None else None
            ),
            "adaptive_remaining_pf_error_bound": (
                float(self.adaptive_remaining_pf_error) if self.adaptive_remaining_pf_error is not None else None
            ),
            "adaptive_max_depth_reached_intervals": self.adaptive_max_depth_reached_cells,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "total_error_log1p_relative": (
                float(self.estimated_logpf_error) if self.estimated_logpf_error is not None else None
            ),
            "truncation_pf_error_bound": (
                float(self.truncation_pf_error_bound) if self.truncation_pf_error_bound is not None else None
            ),
            "u_integration_bounds": list(self.u_bounds_used) if self.u_bounds_used is not None else None,
        }

        data: dict[str, object] = {
            "failure_probability": float(self.pf),
            "design_point_reliability_index": float(self.beta_star),
            "pf_reliability_index": float(self.beta_pf),
            "alpha": self.alpha.tolist(),
            "integration_diagnostics": diagnostics,
        }
        if self.design_point_u is not None:
            data["design_point_u"] = self.design_point_u.to_dict()
        if self.design_point_physical is not None:
            data["design_point_physical"] = self.design_point_physical.to_dict()
        if self.hazard_level is not None:
            data["hazard_level"] = float(self.hazard_level)
        if include_samples:
            data["failure_probability_weights"] = self.failure_samples.weights.tolist()
            data["failure_points_u"] = self.failure_samples.points.tolist()
            if self.failure_samples.hazard_levels is not None:
                data["failure_hazard_levels"] = self.failure_samples.hazard_levels.tolist()
            if self.failure_samples.solicitation_levels is not None:
                data["failure_solicitation_levels"] = self.failure_samples.solicitation_levels.tolist()
        if include_trace and self.failure_samples.integration_trace is not None:
            data["integration_interval_trace"] = self.failure_samples.integration_trace.to_user_dict(
                include_legacy_names=include_legacy_names
            )
        if include_legacy_names:
            data["legacy"] = self._to_legacy_dict(include_samples=include_samples, include_trace=include_trace)
        return data

    def _to_legacy_dict(self, include_samples: bool = False, include_trace: bool = False) -> dict[str, object]:
        """Serialize with the legacy compact key schema.

        Parameters
        ----------
        include_samples : bool
            Whether to include failure sample arrays.
        include_trace : bool
            Whether to include interval diagnostics trace.

        Returns
        -------
        dict[str, object]
            JSON-serializable payload in the legacy key schema.
        """
        data: dict[str, object] = {
            "pf": float(self.pf),
            "beta_star": float(self.beta_star),
            "beta_pf": float(self.beta_pf),
            "alpha": self.alpha.tolist(),
            "diagnostics": {
                "coarse_fail_cells": self.failure_samples.coarse_fail_cells,
                "refined_fail_cells": self.failure_samples.refined_fail_cells,
                "mixed_cells": self.failure_samples.mixed_cells,
                "adaptive_converged": self.adaptive_converged,
                "adaptive_iterations": self.adaptive_iterations,
                "adaptive_estimated_logpf_error": (
                    float(self.adaptive_estimated_logpf_error)
                    if self.adaptive_estimated_logpf_error is not None
                    else None
                ),
                "adaptive_remaining_pf_error": (
                    float(self.adaptive_remaining_pf_error) if self.adaptive_remaining_pf_error is not None else None
                ),
                "adaptive_max_depth_reached_cells": self.adaptive_max_depth_reached_cells,
                "converged": self.converged,
                "convergence_reason": self.convergence_reason,
                "estimated_logpf_error": (
                    float(self.estimated_logpf_error) if self.estimated_logpf_error is not None else None
                ),
                "truncation_pf_error_bound": (
                    float(self.truncation_pf_error_bound) if self.truncation_pf_error_bound is not None else None
                ),
                "u_bounds_used": list(self.u_bounds_used) if self.u_bounds_used is not None else None,
            },
        }
        if self.design_point_u is not None:
            data["design_point_u"] = self.design_point_u.to_dict()
        if self.design_point_physical is not None:
            data["design_point_physical"] = self.design_point_physical.to_dict()
        if self.hazard_level is not None:
            data["hazard_level"] = float(self.hazard_level)
        if include_samples:
            data["failure_weights"] = self.failure_samples.weights.tolist()
            data["failure_points"] = self.failure_samples.points.tolist()
            if self.failure_samples.hazard_levels is not None:
                data["failure_hazards"] = self.failure_samples.hazard_levels.tolist()
            if self.failure_samples.solicitation_levels is not None:
                data["failure_water_levels"] = self.failure_samples.solicitation_levels.tolist()
        if include_trace and self.failure_samples.integration_trace is not None:
            data["integration_trace"] = self.failure_samples.integration_trace.to_dict()
        return data

    def failure_solicitation_levels(self) -> np.ndarray | None:
        """Return physical solicitation values for failure samples."""
        return self.failure_samples.solicitation_levels

    def __repr__(self) -> str:
        """Concise textual representation for logging and debugging."""
        hazard_str = f", hazard_level={self.hazard_level:.3f}" if self.hazard_level is not None else ""
        adaptive_str = ""
        if self.adaptive_converged is not None:
            adaptive_err = (
                f"{self.adaptive_estimated_logpf_error:.3e}"
                if self.adaptive_estimated_logpf_error is not None
                else "None"
            )
            adaptive_str = (
                f", adaptive_converged={self.adaptive_converged}, "
                f"adaptive_err={adaptive_err}, "
                f"adaptive_iterations={self.adaptive_iterations}"
            )
        converged_str = f", converged={self.converged}" if self.converged is not None else ""
        reason_str = f", reason={self.convergence_reason}" if self.convergence_reason is not None else ""
        trunc_str = (
            f", trunc_pf_err={self.truncation_pf_error_bound:.3e}" if self.truncation_pf_error_bound is not None else ""
        )
        bounds_str = f", u_bounds={self.u_bounds_used}" if self.u_bounds_used is not None else ""
        return (
            "IntegrationResult("
            f"pf={self.pf:.6f}, "
            f"beta_star={self.beta_star:.3f}, "
            f"beta_pf={self.beta_pf:.3f}, "
            f"alpha={[round(a, 3) for a in self.alpha.tolist()]}"
            f"{hazard_str}"
            f"{converged_str}"
            f"{reason_str}"
            f"{trunc_str}"
            f"{bounds_str}"
            f"{adaptive_str})"
        )
