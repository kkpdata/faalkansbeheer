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

    def __repr__(self) -> str:
        """Summarize weights and diagnostics."""
        return (
            "FailureSamples("
            f"count={self.weights.size}, "
            f"coarse_fail_cells={self.coarse_fail_cells}, "
            f"refined_fail_cells={self.refined_fail_cells}, "
            f"mixed_cells={self.mixed_cells})"
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

    def summary(self) -> dict[str, float | str | list[float] | None]:
        """Return the headline metrics in a compact dictionary."""
        return {
            "pf": float(self.pf),
            "beta_star": float(self.beta_star),
            "beta_pf": float(self.beta_pf),
            "alpha": self.alpha.tolist(),
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
        return (
            "IntegrationResult("
            f"pf={self.pf:.6f}, "
            f"beta_star={self.beta_star:.3f}, "
            f"beta_pf={self.beta_pf:.3f}, "
            f"alpha={[round(a, 3) for a in self.alpha.tolist()]}"
            f"{hazard_str})"
        )
