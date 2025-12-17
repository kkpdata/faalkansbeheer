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
    """
    Coordinate grid integration in either distribution or hazard/fragility mode.

    Parameters
    ----------
    config : IntegrationConfig
        User-supplied configuration describing the distributions/curves, grid settings, and U-range.
    """

    def __init__(self, config: IntegrationConfig) -> None:
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
        x : np.ndarray | float
            Values in U-space where the CDF is evaluated.

        Returns
        -------
        np.ndarray | float
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
        u1 : np.ndarray
            U-space coordinates for resistance.
        u2 : np.ndarray
            U-space coordinates for solicitation.

        Returns
        -------
        np.ndarray
            Limit-state values with the same broadcastable shape.

        Raises
        ------
        ValueError
            If ``u1`` and ``u2`` do not share the same shape.
        """
        u1_arr = np.asarray(u1)
        u2_arr = np.asarray(u2)
        if u1_arr.shape != u2_arr.shape:
            raise ValueError("u1 and u2 must share the same shape for evaluation.")

        r_vals = self._map_u_to_distribution(u1_arr, self.r_distribution)
        s_vals = self._map_u_to_distribution(u2_arr, self.s_distribution)

        return r_vals - s_vals

    def _map_u_to_distribution(
        self,
        u_values: np.ndarray,
        dist: ot.Distribution,
        u_values_cdf: np.ndarray | None = None,
        u_values_survival: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map U-space coordinates to distribution quantiles.

        Parameters
        ----------
        u_values : np.ndarray
            Coordinates in standard-normal space.
        dist : ot.Distribution
            Distribution whose quantiles will be evaluated.
        u_values_cdf : np.ndarray | None, optional
            Precomputed lower-tail probabilities for ``u_values``. When omitted,
            they are derived internally.
        u_values_survival : np.ndarray | None, optional
            Precomputed survival probabilities (upper tail) for ``u_values``.

        Returns
        -------
        np.ndarray
            Quantiles with the same shape as ``u_values``.
        """
        arr = np.asarray(u_values)
        need_cdf = u_values_cdf is None
        need_survival = u_values_survival is None
        if need_cdf or need_survival:
            cdf_vals, survival_vals = self._normal_probabilities(
                arr,
                compute_cdf=need_cdf,
                compute_survival=need_survival,
            )
            if need_cdf:
                u_values_cdf = cdf_vals
            if need_survival:
                u_values_survival = survival_vals

        flat_u = arr.reshape(-1)
        flat_cdf = np.asarray(u_values_cdf).reshape(-1)
        flat_survival = np.asarray(u_values_survival).reshape(-1)
        quantiles = np.empty_like(flat_cdf, dtype=float)

        lower_mask = flat_u <= 0
        upper_mask = ~lower_mask

        if np.any(lower_mask):
            lower_vals = np.array(dist.computeQuantile(flat_cdf[lower_mask])).flatten()
            quantiles[lower_mask] = lower_vals

        if np.any(upper_mask):
            upper_vals = np.array(dist.computeQuantile(flat_survival[upper_mask], True)).flatten()
            quantiles[upper_mask] = upper_vals

        return quantiles.reshape(arr.shape)

    def _normal_probabilities(
        self,
        u_values: np.ndarray,
        compute_cdf: bool = True,
        compute_survival: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return lower-tail CDF and survival probabilities.

        Parameters
        ----------
        u_values : np.ndarray
            Standard-normal coordinates.
        compute_cdf : bool, default=True
            Whether to compute the lower-tail probabilities.
        compute_survival : bool, default=True
            Whether to compute the upper-tail probabilities.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Lower-tail CDF values and survival probabilities (each may be
            ``None`` when the corresponding ``compute_*`` flag is ``False``).
        """
        arr = np.asarray(u_values)
        flat = arr.reshape(-1)
        cdf = None
        survival = None

        if compute_cdf:
            cdf_vals = np.array(self.std_normal.computeCDF(flat[:, np.newaxis])).flatten()
            cdf = cdf_vals.reshape(arr.shape)

        if compute_survival:
            surv_vals = np.array(self.std_normal.computeCDF((-flat)[:, np.newaxis])).flatten()
            survival = surv_vals.reshape(arr.shape)

        return cdf, survival

    def _u_interval_probabilities(
        self,
        edges: np.ndarray,
        edges_cdf: np.ndarray | None = None,
        edges_survival: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute standard-normal probabilities for intervals defined by ``edges``.

        Parameters
        ----------
        edges : np.ndarray
            Monotone U-grid coordinates whose consecutive pairs define intervals.
        edges_cdf : np.ndarray | None, optional
            Lower-tail CDF values for ``edges``. Computed on demand when omitted.
        edges_survival : np.ndarray | None, optional
            Survival probabilities for ``edges``. Computed on demand when omitted.

        Returns
        -------
        np.ndarray
            Interval probabilities matching ``edges.shape[:-1]``.
        """
        arr = np.asarray(edges)
        need_cdf = edges_cdf is None
        need_survival = edges_survival is None
        if need_cdf or need_survival:
            computed_cdf, computed_survival = self._normal_probabilities(
                arr,
                compute_cdf=need_cdf,
                compute_survival=need_survival,
            )
            if need_cdf:
                edges_cdf = computed_cdf
            if need_survival:
                edges_survival = computed_survival

        left = arr[..., :-1]
        right = arr[..., 1:]
        cdf_left = edges_cdf[..., :-1]
        cdf_right = edges_cdf[..., 1:]
        surv_left = edges_survival[..., :-1]
        surv_right = edges_survival[..., 1:]

        probs = np.empty_like(left, dtype=float)

        neg_mask = right <= 0
        pos_mask = left >= 0
        cross_mask = ~(neg_mask | pos_mask)

        if np.any(neg_mask):
            probs[neg_mask] = cdf_right[neg_mask] - cdf_left[neg_mask]
        if np.any(pos_mask):
            probs[pos_mask] = surv_left[pos_mask] - surv_right[pos_mask]
        if np.any(cross_mask):
            zero_cdf = 0.5
            zero_survival = 0.5
            neg_part = zero_cdf - cdf_left[cross_mask]
            pos_part = zero_survival - surv_right[cross_mask]
            probs[cross_mask] = neg_part + pos_part

        return probs

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

        u_edges_cdf, u_edges_survival = self._normal_probabilities(u_edges)
        u_contrib = self._u_interval_probabilities(u_edges, u_edges_cdf, u_edges_survival)
        total_contrib = float(u_contrib.sum())
        if total_contrib <= 0.0:
            raise RuntimeError("Invalid U-grid configuration produced non-positive probability mass.")
        u_contrib = u_contrib / total_contrib

        r_edges = self._map_u_to_distribution(
            u_edges,
            self.r_distribution,
            u_values_cdf=u_edges_cdf,
            u_values_survival=u_edges_survival,
        )
        s_edges = self._map_u_to_distribution(
            u_edges,
            self.s_distribution,
            u_values_cdf=u_edges_cdf,
            u_values_survival=u_edges_survival,
        )

        # For monotonically increasing quantile mappings, each cell spans [edge_i, edge_{i+1}]
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

            cdf_u1_sub, surv_u1_sub = self._normal_probabilities(u1_sub_edges)
            cdf_u2_sub, surv_u2_sub = self._normal_probabilities(u2_sub_edges)

            sub_prob_u1 = self._u_interval_probabilities(u1_sub_edges, cdf_u1_sub, surv_u1_sub)
            sub_prob_u2 = self._u_interval_probabilities(u2_sub_edges, cdf_u2_sub, surv_u2_sub)
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
        u_values : np.ndarray
            Points along the R-axis (in U-space) used to trace the limit-state curve.

        Returns
        -------
        np.ndarray
            Matching U-space values for the solicitation axis that satisfy ``r - s = 0``.
        """
        u_values = np.asarray(u_values, dtype=float)
        cdf_u, survival_u = self._normal_probabilities(u_values, compute_cdf=True, compute_survival=True)
        r_vals = self._map_u_to_distribution(
            u_values,
            self.r_distribution,
            u_values_cdf=cdf_u,
            u_values_survival=survival_u,
        )

        s_cdf = np.array(self.s_distribution.computeCDF(r_vals[:, np.newaxis])).flatten()
        s_survival = np.array(self.s_distribution.computeSurvivalFunction(r_vals[:, np.newaxis])).flatten()
        s_cdf = np.clip(s_cdf, 1e-300, 1)
        s_survival = np.clip(s_survival, 1e-300, 1)

        u2_vals = np.empty_like(s_cdf, dtype=float)
        lower_mask = s_cdf <= 0.5

        if np.any(lower_mask):
            lprobs = s_cdf[lower_mask]
            u2_vals[lower_mask] = np.array(self.std_normal.computeQuantile(lprobs)).reshape(lprobs.shape)

        if np.any(~lower_mask):
            uprobs = s_survival[~lower_mask]
            u2_vals[~lower_mask] = np.array(self.std_normal.computeQuantile(uprobs, True)).reshape(uprobs.shape)

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
        samples : FailureSamples
            Weighted failure cell representation created by the integration step.

        Returns
        -------
        IntegrationResult
            Failure probability, design point, and diagnostics.

        Raises
        ------
        RuntimeError
            If no failure cells are available for post-processing.
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
