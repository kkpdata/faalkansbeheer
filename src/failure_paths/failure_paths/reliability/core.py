from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openturns as ot

from .config import IntegrationConfig
from .curve_distributions import FragilityDerivedDistribution, HazardDerivedDistribution
from .results import FailureSamples, IntegrationResult


@dataclass
class _DistributionGridData:
    """Helper container describing the discretized U-grid."""

    u_edges: np.ndarray
    u_centers: np.ndarray
    cell_weights: np.ndarray
    fail_mask: np.ndarray
    safe_mask: np.ndarray
    mixed_mask: np.ndarray
    u1_center_grid: np.ndarray
    u2_center_grid: np.ndarray
    mixed_indices: np.ndarray
    refine_factor: int
    subcell_weights: np.ndarray | None = None
    subcell_fail_mask: np.ndarray | None = None
    u1_sub_mesh: np.ndarray | None = None
    u2_sub_mesh: np.ndarray | None = None
    u1_sub_edges: np.ndarray | None = None
    u2_sub_edges: np.ndarray | None = None


class ReliabilityIntegrator:
    """Coordinate grid integration in either distribution or hazard/fragility mode."""

    def __init__(self, config: IntegrationConfig) -> None:
        """Store configuration for later runs."""
        self.config = config
        self._using_curve_distributions = False
        self._r_distribution: ot.Distribution
        self._s_distribution: ot.Distribution
        self._initialize_distributions()

    @property
    def r_distribution(self) -> ot.Distribution:
        """Resistance distribution used for R."""
        return self._r_distribution

    @property
    def s_distribution(self) -> ot.Distribution:
        """Solicitation distribution used for S."""
        return self._s_distribution

    @property
    def std_normal(self) -> ot.Normal:
        """Shared standard normal distribution used for U-space conversions."""
        return self.config.std_normal

    def standard_normal_cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the standard normal CDF for scalar or array inputs.

        Parameters
        ----------
        x:
            Values in U-space where the CDF is evaluated.

        Returns
        -------
        numpy.ndarray | float
            Probability values with the same shape as the input.
        """
        arr = np.atleast_1d(x)
        values = np.array(self.std_normal.computeCDF(arr[:, np.newaxis])).flatten()
        if np.isscalar(x):
            return values[0]
        return values

    def evaluate_limit_state(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
        """Evaluate `g(R, S) = R - S` over the provided U-grid.

        Parameters
        ----------
        u1, u2:
            Arrays containing the U-coordinates for R and S.

        Returns
        -------
        numpy.ndarray
            Limit-state values with the same broadcastable shape.
        """
        u1_arr = np.asarray(u1)
        u2_arr = np.asarray(u2)
        if u1_arr.shape != u2_arr.shape:
            raise ValueError("u1 and u2 must share the same shape for evaluation.")

        probs_r = self.standard_normal_cdf(u1_arr.reshape(-1))
        probs_s = self.standard_normal_cdf(u2_arr.reshape(-1))

        r_vals = np.array(self.r_distribution.computeQuantile(probs_r)).flatten()
        s_vals = np.array(self.s_distribution.computeQuantile(probs_s)).flatten()

        return (r_vals - s_vals).reshape(u1_arr.shape)

    def run(self) -> IntegrationResult:
        """Execute the integration workflow and post-process failure samples."""
        samples = self.integrate_failure_samples()
        return self._postprocess_failure_samples(samples)

    def integrate_failure_samples(self) -> FailureSamples:
        """Generate failure point samples using the configured U-grid."""
        samples = self._integrate_distributions()
        if self._using_curve_distributions and samples.points.size > 0 and self.config.hazard_curve is not None:
            beta_vals = samples.points[:, 1]
            samples.hazard_levels = self.config.hazard_curve.hazard_from_beta(beta_vals)
        return samples

    def _compute_distribution_grid(self) -> _DistributionGridData:
        """Assemble the U-grid, masks, and refinement info for distribution mode.

        Returns
        -------
        _DistributionGridData
            Cached coordinates, probabilities, and mask arrays describing the grid.
        """
        cfg = self.config
        u_edges = np.linspace(cfg.u_min, cfg.u_max, cfg.coarse_points)
        u_centers = 0.5 * (u_edges[1:] + u_edges[:-1])
        u_edges_cdf = self.standard_normal_cdf(u_edges)
        u_contrib = np.diff(u_edges_cdf)

        r_edges = np.array(self.r_distribution.computeQuantile(u_edges_cdf)).flatten()
        s_edges = np.array(self.s_distribution.computeQuantile(u_edges_cdf)).flatten()

        r_min = r_edges[:-1][:, None]
        r_max = r_edges[1:][:, None]
        s_min = s_edges[:-1][None, :]
        s_max = s_edges[1:][None, :]

        z_min = r_min - s_max
        z_max = r_max - s_min

        fail_mask = z_max <= 0.0
        safe_mask = z_min >= 0.0
        mixed_mask = ~(fail_mask | safe_mask)

        u1_center_grid, u2_center_grid = np.meshgrid(u_centers, u_centers, indexing="ij")
        cell_weights = u_contrib[:, None] * u_contrib[None, :]

        mixed_idx = np.argwhere(mixed_mask)
        refine_factor = cfg.refine_factor

        sub_weights = None
        sub_fail = None
        u1_sub_mesh = None
        u2_sub_mesh = None
        u1_sub_edges = None
        u2_sub_edges = None

        if mixed_idx.size > 0 and refine_factor > 0:
            frac_edges = np.linspace(0.0, 1.0, refine_factor + 1)
            frac_centers = 0.5 * (frac_edges[1:] + frac_edges[:-1])

            u1_left = u_edges[:-1][mixed_idx[:, 0]]
            u1_right = u_edges[1:][mixed_idx[:, 0]]
            u2_left = u_edges[:-1][mixed_idx[:, 1]]
            u2_right = u_edges[1:][mixed_idx[:, 1]]

            u1_sub_edges = u1_left[:, None] + (u1_right - u1_left)[:, None] * frac_edges
            u2_sub_edges = u2_left[:, None] + (u2_right - u2_left)[:, None] * frac_edges

            cdf_u1_sub = self.standard_normal_cdf(u1_sub_edges.reshape(-1)).reshape(-1, refine_factor + 1)
            cdf_u2_sub = self.standard_normal_cdf(u2_sub_edges.reshape(-1)).reshape(-1, refine_factor + 1)

            sub_prob_u1 = np.diff(cdf_u1_sub, axis=1)
            sub_prob_u2 = np.diff(cdf_u2_sub, axis=1)
            sub_weights = sub_prob_u1[:, :, None] * sub_prob_u2[:, None, :]

            u1_sub_centers = u1_left[:, None] + (u1_right - u1_left)[:, None] * frac_centers
            u2_sub_centers = u2_left[:, None] + (u2_right - u2_left)[:, None] * frac_centers

            u1_sub_mesh = np.broadcast_to(
                u1_sub_centers[:, :, None],
                (mixed_idx.shape[0], refine_factor, refine_factor),
            )
            u2_sub_mesh = np.broadcast_to(
                u2_sub_centers[:, None, :],
                (mixed_idx.shape[0], refine_factor, refine_factor),
            )

            z_sub = self.evaluate_limit_state(u1_sub_mesh, u2_sub_mesh)
            sub_fail = z_sub <= 0.0

        return _DistributionGridData(
            u_edges=u_edges,
            u_centers=u_centers,
            cell_weights=cell_weights,
            fail_mask=fail_mask,
            safe_mask=safe_mask,
            mixed_mask=mixed_mask,
            u1_center_grid=u1_center_grid,
            u2_center_grid=u2_center_grid,
            mixed_indices=mixed_idx,
            refine_factor=refine_factor,
            subcell_weights=sub_weights,
            subcell_fail_mask=sub_fail,
            u1_sub_mesh=u1_sub_mesh,
            u2_sub_mesh=u2_sub_mesh,
            u1_sub_edges=u1_sub_edges,
            u2_sub_edges=u2_sub_edges,
        )

    def _limit_state_curve(self, u_values: np.ndarray) -> np.ndarray:
        """Return U-space coordinates of the z=0 curve for supplied R-axis samples.

        Parameters
        ----------
        u_values : numpy.ndarray
            Points along the R-axis (in U-space) used to trace the limit-state curve.

        Returns
        -------
        numpy.ndarray
            Matching U-space values for the solicitation axis that satisfy ``r - s = 0``.
        """
        probs_r = self.standard_normal_cdf(u_values)
        r_vals = np.array(self.r_distribution.computeQuantile(probs_r)).flatten()
        s_cdf = np.array(self.s_distribution.computeCDF(r_vals[:, np.newaxis])).flatten()
        u2_vals = np.array(self.std_normal.computeQuantile(s_cdf)).flatten()
        return u2_vals

    # Distribution-based implementation
    def _integrate_distributions(self) -> FailureSamples:
        """Integrate probability mass over the U-grid in distribution space.

        Returns
        -------
        FailureSamples
            Weighted failure cells after optional refinement.
        """
        grid = self._compute_distribution_grid()

        w_fail_coarse = grid.cell_weights[grid.fail_mask]
        u1_fail_coarse = grid.u1_center_grid[grid.fail_mask]
        u2_fail_coarse = grid.u2_center_grid[grid.fail_mask]

        w_fail_refined: list[np.ndarray] = []
        u1_fail_refined: list[np.ndarray] = []
        u2_fail_refined: list[np.ndarray] = []
        refined_fail_cells = 0

        if (
            grid.mixed_indices.size > 0
            and grid.refine_factor > 0
            and grid.subcell_fail_mask is not None
            and grid.subcell_weights is not None
            and grid.u1_sub_mesh is not None
            and grid.u2_sub_mesh is not None
        ):
            fail_sub = grid.subcell_fail_mask
            refined_fail_cells = int(np.count_nonzero(fail_sub))

            if np.any(fail_sub):
                w_fail_refined.append(grid.subcell_weights[fail_sub])
                u1_fail_refined.append(grid.u1_sub_mesh[fail_sub])
                u2_fail_refined.append(grid.u2_sub_mesh[fail_sub])

        weight_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []

        if w_fail_coarse.size > 0:
            weight_parts.append(w_fail_coarse)
            point_parts.append(np.column_stack([u1_fail_coarse, u2_fail_coarse]))

        if w_fail_refined:
            weight_parts.append(np.concatenate(w_fail_refined))
            point_parts.append(np.column_stack([np.concatenate(u1_fail_refined), np.concatenate(u2_fail_refined)]))

        if weight_parts:
            weights = np.concatenate(weight_parts)
            points = np.vstack(point_parts)
        else:
            weights = np.array([])
            points = np.empty((0, 2))

        return FailureSamples(
            weights=weights,
            points=points,
            coarse_fail_cells=int(np.count_nonzero(grid.fail_mask)),
            refined_fail_cells=refined_fail_cells,
            mixed_cells=int(grid.mixed_indices.shape[0]),
            hazard_levels=None,
        )

    def _initialize_distributions(self) -> None:
        """Resolve the actual R/S distributions, honoring curve inputs when provided."""
        cfg = self.config
        self._using_curve_distributions = False
        if cfg.fragility_curve is not None:
            self._r_distribution = FragilityDerivedDistribution(cfg.fragility_curve)
            self._using_curve_distributions = True
        else:
            if cfg.r_distribution is None:
                raise ValueError("r_distribution is required when no fragility_curve is provided.")
            self._r_distribution = cfg.r_distribution

        if cfg.hazard_curve is not None:
            self._s_distribution = HazardDerivedDistribution(cfg.hazard_curve)
            self._using_curve_distributions = True
        else:
            if cfg.s_distribution is None:
                raise ValueError("s_distribution is required when no hazard_curve is provided.")
            self._s_distribution = cfg.s_distribution

    def _postprocess_failure_samples(self, samples: FailureSamples) -> IntegrationResult:
        """Compute reliability metrics from weighted failure points.

        Parameters
        ----------
        samples:
            Weighted failure cell representation created by the integration step.

        Returns
        -------
        IntegrationResult
            Failure probability, design point, and diagnostics.
        """
        w_fail = samples.weights
        U_fail = samples.points
        if w_fail.size == 0:
            raise RuntimeError("No failure cells detected during refinement.")

        pf = float(w_fail.sum())
        beta_pf = float(self.std_normal.computeQuantile(pf, True)[0])

        udist_fail = np.linalg.norm(U_fail, axis=1)
        near_idx = int(np.argmin(udist_fail))
        beta_star = float(udist_fail[near_idx])
        u_vec = U_fail[near_idx]
        alpha_val = -u_vec / beta_star

        hazard_level = None
        if self.config.hazard_curve is not None:
            hazard_level = float(self.config.hazard_curve.hazard_from_beta(beta_star))

        return IntegrationResult(
            pf=pf,
            beta_star=beta_star,
            alpha=alpha_val,
            failure_samples=samples,
            hazard_level=hazard_level,
            beta_pf=beta_pf,
        )
