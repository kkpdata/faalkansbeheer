from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FailureSamples:
    """Weighted failure points identified by the integration scheme."""

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

    def __repr__(self) -> str:
        """Summarize weights and diagnostics."""
        adaptive_bits = ""
        if self.adaptive_converged is not None:
            adaptive_bits = (
                f", adaptive_converged={self.adaptive_converged}, "
                f"adaptive_iterations={self.adaptive_iterations}, "
                f"adaptive_max_depth_cells={self.adaptive_max_depth_reached_cells}"
            )
        return (
            "FailureSamples("
            f"count={self.weights.size}, "
            f"coarse_fail_cells={self.coarse_fail_cells}, "
            f"refined_fail_cells={self.refined_fail_cells}, "
            f"mixed_cells={self.mixed_cells}"
            f"{adaptive_bits})"
        )


@dataclass
class IntegrationResult:
    """Summary of reliability metrics produced by the grid integrator."""

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

    def summary(self) -> dict[str, float | str | list[float] | None]:
        """Return the headline metrics in a compact dictionary."""
        return {
            "pf": float(self.pf),
            "beta_star": float(self.beta_star),
            "beta_pf": float(self.beta_pf),
            "alpha": self.alpha.tolist(),
            "adaptive_converged": self.adaptive_converged,
            "adaptive_estimated_logpf_error": (
                float(self.adaptive_estimated_logpf_error) if self.adaptive_estimated_logpf_error is not None else None
            ),
        }

    def to_dict(self, include_samples: bool = False) -> dict[str, object]:
        """Serialize the full integration result.

        Parameters
        ----------
        include_samples : bool
            Whether to include the failure samples and weights.

        Returns
        -------
        dict[str, object]
            JSON-serializable structure containing the result summary and optional samples.
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
            },
        }
        if self.hazard_level is not None:
            data["hazard_level"] = float(self.hazard_level)
        if include_samples:
            data["failure_weights"] = self.failure_samples.weights.tolist()
            data["failure_points"] = self.failure_samples.points.tolist()
            if self.failure_samples.hazard_levels is not None:
                data["failure_hazards"] = self.failure_samples.hazard_levels.tolist()
            if self.failure_samples.solicitation_levels is not None:
                data["failure_water_levels"] = self.failure_samples.solicitation_levels.tolist()
        return data

    def failure_water_levels(self) -> np.ndarray | None:
        """Return physical solicitation (water level) values for failure samples."""
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
        return (
            "IntegrationResult("
            f"pf={self.pf:.6f}, "
            f"beta_star={self.beta_star:.3f}, "
            f"beta_pf={self.beta_pf:.3f}, "
            f"alpha={[round(a, 3) for a in self.alpha.tolist()]}"
            f"{hazard_str}"
            f"{adaptive_str})"
        )
