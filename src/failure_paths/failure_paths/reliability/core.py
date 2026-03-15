from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import openturns as ot

from ..common.prob import beta_from_pf, pf_from_beta
from .config import IntegrationConfig
from .curve_distributions import FragilityDerivedDistribution, HazardDerivedDistribution
from .results import FailureSamples, IntegrationResult


@dataclass
class _DistributionGridData:
    """Helper container describing the discretized U-grid."""

    u_min: float
    u_max: float
    u1_edges: np.ndarray
    u2_edges: np.ndarray
    u1_centers: np.ndarray
    u2_centers: np.ndarray
    cell_weights: np.ndarray
    fail_mask: np.ndarray
    safe_mask: np.ndarray
    mixed_mask: np.ndarray
    u1_center_grid: np.ndarray
    u2_center_grid: np.ndarray
    mixed_indices: np.ndarray
    captured_u1_mass: float
    captured_u2_mass: float
    captured_domain_mass: float
    target_domain_mass: float


@dataclass
class _AdaptiveCell:
    """Adaptive mixed-cell state used for local error-driven refinement."""

    u1_left: float
    u1_right: float
    u2_left: float
    u2_right: float
    weight: float
    depth: int
    center_u1: float
    center_u2: float
    mass_est: float
    error_est: float
    fail_estimate: bool
    refinable: bool


@dataclass
class _BoundsPassResult:
    """Container for one integration pass at fixed U-space bounds."""

    samples: FailureSamples
    truncation_pf_error_bound: float
    total_pf_error_bound: float
    estimated_logpf_error: float
    converged: bool
    u_bounds_used: tuple[float, float]


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

    _AUTO_BOUNDS_WIDEN_STEP = 1.0
    _AUTO_BOUNDS_MAX_ABS = 16.0
    _MAX_DISTRIBUTION_STEP_LEVELS = 512
    _CURVE_LEVEL_MIN_SPACING_X = 1.0e-4
    _MIN_U_EDGE_SPACING = 1.0e-10

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
        values = pf_from_beta(x, tail="lower")
        if np.isscalar(x):
            return float(values)
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
        compute_cdf : bool
            Whether to compute the lower-tail probabilities. Defaults to ``True``.
        compute_survival : bool
            Whether to compute the upper-tail probabilities. Defaults to ``True``.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None]
            Lower-tail CDF values and survival probabilities (each may be
            ``None`` when the corresponding ``compute_*`` flag is ``False``).
        """
        arr = np.asarray(u_values)
        cdf = None
        survival = None

        if compute_cdf:
            cdf = pf_from_beta(arr, tail="lower")

        if compute_survival:
            survival = pf_from_beta(arr, tail="upper")

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
        if samples.points.size > 0:
            u_s = samples.points[:, 1]
            cdf_vals, surv_vals = self._normal_probabilities(u_s, compute_cdf=True, compute_survival=True)
            solicitation_levels = self._map_u_to_distribution(
                u_s,
                self.s_distribution,
                u_values_cdf=cdf_vals,
                u_values_survival=surv_vals,
            )
            samples.solicitation_levels = solicitation_levels
        if self._using_curve_distributions and samples.points.size > 0 and self.config.hazard_curve is not None:
            beta_vals = samples.points[:, 1]
            samples.hazard_levels = self.config.hazard_curve.hazard_from_beta(beta_vals)
        return samples

    def _initial_u_bounds(self) -> tuple[float, float]:
        """Resolve initial U-space integration bounds from config."""
        if self.config.u_manual_bounds is not None:
            lower, upper = self.config.u_manual_bounds
            return float(lower), float(upper)

        two_sided_tail = float(self.config.u_tail_probability)
        half_tail = 0.5 * two_sided_tail
        bound = abs(float(beta_from_pf(half_tail, tail="lower")))
        if not np.isfinite(bound) or bound <= 0.0:
            raise RuntimeError("Failed to derive finite automatic U-space bounds from u_tail_probability.")
        return -bound, bound

    def _widen_bounds(self, bounds: tuple[float, float]) -> tuple[float, float] | None:
        """Expand U-space bounds by a fixed step, capped at configured hard limits."""
        lower, upper = float(bounds[0]), float(bounds[1])
        widened_lower = max(-self._AUTO_BOUNDS_MAX_ABS, lower - self._AUTO_BOUNDS_WIDEN_STEP)
        widened_upper = min(self._AUTO_BOUNDS_MAX_ABS, upper + self._AUTO_BOUNDS_WIDEN_STEP)
        if widened_lower == lower and widened_upper == upper:
            return None
        return widened_lower, widened_upper

    def _truncation_pf_error_bound(self, grid: _DistributionGridData) -> float:
        """Compute missing probability mass outside the represented integration domain."""
        return max(0.0, float(grid.target_domain_mass - grid.captured_domain_mass))

    @staticmethod
    def _merge_level_candidates(*level_groups: np.ndarray | list[float] | None) -> np.ndarray:
        """Merge candidate physical levels into a finite sorted unique array."""
        merged: list[np.ndarray] = []
        for group in level_groups:
            if group is None:
                continue
            arr = np.asarray(group, dtype=float).reshape(-1)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            merged.append(arr)
        if not merged:
            return np.array([], dtype=float)
        return np.sort(np.unique(np.concatenate(merged)))

    @staticmethod
    def _deduplicate_with_min_spacing(values: np.ndarray | list[float], min_spacing: float) -> np.ndarray:
        """Sort and deduplicate values while enforcing a minimum spacing."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = np.sort(arr)
        if min_spacing <= 0.0:
            return np.unique(arr)

        kept = [float(arr[0])]
        last = float(arr[0])
        for value in arr[1:]:
            value_f = float(value)
            if value_f - last >= min_spacing:
                kept.append(value_f)
                last = value_f
        return np.asarray(kept, dtype=float)

    def _clip_levels_to_range(
        self,
        levels: np.ndarray | list[float] | None,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        """Clip candidate physical levels to finite values inside [x_min, x_max]."""
        if levels is None:
            return np.array([], dtype=float)
        arr = np.asarray(levels, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = arr[(arr >= x_min) & (arr <= x_max)]
        if arr.size == 0:
            return np.array([], dtype=float)
        return self._deduplicate_with_min_spacing(arr, self._CURVE_LEVEL_MIN_SPACING_X)

    def _distribution_value_bounds(self, dist: ot.Distribution, u_min: float, u_max: float) -> tuple[float, float]:
        """Map U-space bounds to finite physical-space bounds for a distribution."""
        u_edges = np.array([u_min, u_max], dtype=float)
        cdf_edges, surv_edges = self._normal_probabilities(u_edges)
        x_edges = self._map_u_to_distribution(
            u_edges,
            dist,
            u_values_cdf=cdf_edges,
            u_values_survival=surv_edges,
        )
        x_edges = np.asarray(x_edges, dtype=float).reshape(-1)
        finite = x_edges[np.isfinite(x_edges)]
        if finite.size == 0:
            raise RuntimeError("Failed to derive finite physical bounds from U-space bounds.")
        return float(np.min(finite)), float(np.max(finite))

    def _distribution_step_levels(self, dist: ot.Distribution, x_min: float, x_max: float) -> np.ndarray:
        """Collect step/singularity level candidates from a distribution."""
        levels: list[float] = []

        # Singularities are available for some mixed distributions.
        try:
            singularities = dist.getSingularities()
            levels.extend(float(singularities[i]) for i in range(singularities.getSize()))
        except Exception:
            pass

        is_discrete = False
        try:
            is_discrete = bool(dist.isDiscrete())
        except Exception:
            is_discrete = False

        if is_discrete:
            support_values: np.ndarray | None = None
            try:
                interval = ot.Interval([float(x_min)], [float(x_max)])
                support = dist.getSupport(interval)
                support_values = np.asarray(support, dtype=float).reshape(-1)
            except Exception:
                try:
                    support = dist.getSupport()
                    support_values = np.asarray(support, dtype=float).reshape(-1)
                except Exception:
                    support_values = None
            if support_values is not None and support_values.size > 0:
                levels.extend(float(v) for v in support_values)

        if not levels:
            return np.array([], dtype=float)

        arr = np.asarray(levels, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = arr[(arr >= x_min) & (arr <= x_max)]
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = np.sort(np.unique(arr))

        if arr.size > self._MAX_DISTRIBUTION_STEP_LEVELS:
            idx = np.linspace(0, arr.size - 1, self._MAX_DISTRIBUTION_STEP_LEVELS)
            arr = arr[np.unique(np.round(idx).astype(int))]
        return arr

    def _compute_distribution_grid(
        self,
        u_min: float | None = None,
        u_max: float | None = None,
    ) -> _DistributionGridData:
        """Assemble the U-grid, masks, and refinement info for distribution mode.

        Parameters
        ----------
        u_min : float | None, optional
            Lower U-space bound for this grid pass. When ``None``, the lower
            bound is resolved from the configured automatic/manual bound policy.
        u_max : float | None, optional
            Upper U-space bound for this grid pass. When ``None``, the upper
            bound is resolved from the configured automatic/manual bound policy.

        Returns
        -------
        _DistributionGridData
            Cached coordinates, probabilities, and mask arrays describing the grid.

        Raises
        ------
        ValueError
            If the resolved bounds are invalid (``u_max <= u_min``).
        RuntimeError
            If the U-grid configuration produces invalid probability mass.
        """
        if u_min is None or u_max is None:
            bounds = self._initial_u_bounds()
            if u_min is None:
                u_min = bounds[0]
            if u_max is None:
                u_max = bounds[1]
        u_min = float(u_min)
        u_max = float(u_max)
        if not (u_max > u_min):
            raise ValueError("u_max must be greater than u_min.")

        cfg = self.config
        u1_curve_levels = cfg.fragility_curve.hazard_levels if cfg.fragility_curve is not None else None
        u2_curve_levels = cfg.hazard_curve.hazard_levels if cfg.hazard_curve is not None else None

        r_x_min, r_x_max = self._distribution_value_bounds(self.r_distribution, u_min=u_min, u_max=u_max)
        s_x_min, s_x_max = self._distribution_value_bounds(self.s_distribution, u_min=u_min, u_max=u_max)
        shared_curve_levels = self._merge_level_candidates(u1_curve_levels, u2_curve_levels)
        shared_curve_levels = self._deduplicate_with_min_spacing(
            shared_curve_levels,
            self._CURVE_LEVEL_MIN_SPACING_X,
        )
        u1_curve_injected = self._clip_levels_to_range(shared_curve_levels, x_min=r_x_min, x_max=r_x_max)
        u2_curve_injected = self._clip_levels_to_range(shared_curve_levels, x_min=s_x_min, x_max=s_x_max)
        r_step_levels = self._distribution_step_levels(self.r_distribution, x_min=r_x_min, x_max=r_x_max)
        s_step_levels = self._distribution_step_levels(self.s_distribution, x_min=s_x_min, x_max=s_x_max)
        shared_step_levels = self._merge_level_candidates(r_step_levels, s_step_levels)
        u1_levels = self._merge_level_candidates(u1_curve_injected, shared_step_levels)
        u2_levels = self._merge_level_candidates(u2_curve_injected, shared_step_levels)

        u1_edges = self._build_axis_edges(
            dist=self.r_distribution,
            curve_levels=u1_levels,
            u_min=u_min,
            u_max=u_max,
        )
        u2_edges = self._build_axis_edges(
            dist=self.s_distribution,
            curve_levels=u2_levels,
            u_min=u_min,
            u_max=u_max,
        )

        u_s_max = None
        target_domain_mass = 1.0
        if cfg.max_solicitation_level is not None:
            prob = float(self.s_distribution.computeCDF(cfg.max_solicitation_level))
            prob = float(np.clip(prob, 0.0, 1.0))
            target_domain_mass = prob
            u_s_max = float(beta_from_pf(prob, tail="lower"))
            if u_min < u_s_max < u_max:
                u2_edges = np.sort(np.unique(np.r_[u2_edges, u_s_max]))

        u1_centers = 0.5 * (u1_edges[1:] + u1_edges[:-1])
        u2_centers = 0.5 * (u2_edges[1:] + u2_edges[:-1])

        u1_edges_cdf, u1_edges_survival = self._normal_probabilities(u1_edges)
        u2_edges_cdf, u2_edges_survival = self._normal_probabilities(u2_edges)
        u1_contrib = self._u_interval_probabilities(u1_edges, u1_edges_cdf, u1_edges_survival)
        u2_contrib = self._u_interval_probabilities(u2_edges, u2_edges_cdf, u2_edges_survival)
        total_contrib_u1 = float(u1_contrib.sum())
        if not np.isfinite(total_contrib_u1) or total_contrib_u1 < 0.0:
            raise RuntimeError("Invalid U-grid configuration produced invalid probability mass.")

        include_mask = None
        if u_s_max is not None:
            if u_s_max <= u2_edges[0]:
                include_mask = np.zeros(u2_edges.size - 1, dtype=bool)
            elif u_s_max >= u2_edges[-1]:
                include_mask = np.ones(u2_edges.size - 1, dtype=bool)
            else:
                include_mask = u2_edges[1:] <= u_s_max
            u2_contrib = np.where(include_mask, u2_contrib, 0.0)
        total_contrib_u2 = float(u2_contrib.sum())
        if not np.isfinite(total_contrib_u2) or total_contrib_u2 < 0.0:
            raise RuntimeError("Invalid U-grid configuration produced invalid probability mass.")

        captured_u1_mass = total_contrib_u1
        captured_u2_mass = total_contrib_u2
        captured_domain_mass = float(captured_u1_mass * captured_u2_mass)

        r_edges = self._map_u_to_distribution(
            u1_edges,
            self.r_distribution,
            u_values_cdf=u1_edges_cdf,
            u_values_survival=u1_edges_survival,
        )
        s_edges = self._map_u_to_distribution(
            u2_edges,
            self.s_distribution,
            u_values_cdf=u2_edges_cdf,
            u_values_survival=u2_edges_survival,
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
        if include_mask is not None:
            column_mask = include_mask[None, :]
            fail_mask = fail_mask & column_mask
            safe_mask = safe_mask & column_mask
            mixed_mask = mixed_mask & column_mask

        u1_center_grid, u2_center_grid = np.meshgrid(u1_centers, u2_centers, indexing="ij")
        cell_weights = u1_contrib[:, None] * u2_contrib[None, :]

        mixed_idx = np.argwhere(mixed_mask)

        return _DistributionGridData(
            u_min=u_min,
            u_max=u_max,
            u1_edges=u1_edges,
            u2_edges=u2_edges,
            u1_centers=u1_centers,
            u2_centers=u2_centers,
            cell_weights=cell_weights,
            fail_mask=fail_mask,
            safe_mask=safe_mask,
            mixed_mask=mixed_mask,
            u1_center_grid=u1_center_grid,
            u2_center_grid=u2_center_grid,
            mixed_indices=mixed_idx,
            captured_u1_mass=captured_u1_mass,
            captured_u2_mass=captured_u2_mass,
            captured_domain_mass=captured_domain_mass,
            target_domain_mass=float(target_domain_mass),
        )

    def _build_axis_edges(
        self,
        dist: ot.Distribution,
        curve_levels: np.ndarray | None,
        u_min: float,
        u_max: float,
    ) -> np.ndarray:
        """Build axis edges from mandatory knot edges plus regular fill-in."""
        cfg = self.config
        mandatory = np.array([u_min, u_max], dtype=float)
        mapped_knots = self._map_curve_levels_to_u_edges(dist=dist, levels=curve_levels, u_min=u_min, u_max=u_max)
        if mapped_knots.size > 0:
            mandatory = np.r_[mandatory, mapped_knots]
        mandatory_edges = self._deduplicate_with_min_spacing(mandatory, self._MIN_U_EDGE_SPACING)

        if mandatory_edges.size >= cfg.coarse_points:
            # Keep all mandatory edges; do not downsample knot-aligned boundaries.
            return mandatory_edges

        n_extra = int(cfg.coarse_points - mandatory_edges.size)
        regular = np.linspace(u_min, u_max, cfg.coarse_points)
        regular_pool = regular[~np.isin(regular, mandatory_edges)]
        if regular_pool.size > 0 and mandatory_edges.size > 0 and self._MIN_U_EDGE_SPACING > 0.0:
            distance_to_mandatory = np.min(np.abs(regular_pool[:, None] - mandatory_edges[None, :]), axis=1)
            regular_pool = regular_pool[distance_to_mandatory >= self._MIN_U_EDGE_SPACING]
        if regular_pool.size == 0 or n_extra <= 0:
            return mandatory_edges

        if n_extra >= regular_pool.size:
            extras = regular_pool
        else:
            # Spread extra regular edges over the available range.
            raw = np.linspace(0, regular_pool.size - 1, n_extra)
            idx = np.unique(np.round(raw).astype(int))
            if idx.size < n_extra:
                missing = n_extra - idx.size
                remaining = np.setdiff1d(np.arange(regular_pool.size), idx, assume_unique=False)
                idx = np.r_[idx, remaining[:missing]]
                idx = np.sort(idx.astype(int))
            extras = regular_pool[idx]

        return np.sort(np.unique(np.r_[mandatory_edges, extras]))

    def _map_curve_levels_to_u_edges(
        self,
        dist: ot.Distribution,
        levels: np.ndarray | None,
        u_min: float,
        u_max: float,
    ) -> np.ndarray:
        """Map curve hazard-level knots to interior U-space edges."""
        if levels is None:
            return np.array([], dtype=float)

        level_arr = np.asarray(levels, dtype=float).reshape(-1)
        if level_arr.size == 0:
            return np.array([], dtype=float)

        level_arr = level_arr[np.isfinite(level_arr)]
        if level_arr.size == 0:
            return np.array([], dtype=float)
        level_arr = self._deduplicate_with_min_spacing(level_arr, self._CURVE_LEVEL_MIN_SPACING_X)
        level_arr = level_arr[np.isfinite(level_arr)]
        if level_arr.size == 0:
            return np.array([], dtype=float)

        cdf = np.array(dist.computeCDF(level_arr[:, np.newaxis])).reshape(-1)
        survival = np.array(dist.computeSurvivalFunction(level_arr[:, np.newaxis])).reshape(-1)
        cdf = np.clip(cdf, 0.0, 1.0)
        survival = np.clip(survival, 0.0, 1.0)

        u_from_cdf = beta_from_pf(cdf, tail="lower")
        u_from_survival = beta_from_pf(survival, tail="upper")
        u_candidates = np.where(np.isfinite(u_from_cdf), u_from_cdf, u_from_survival)
        u_candidates = np.asarray(u_candidates, dtype=float)
        if u_candidates.size == 0:
            return np.array([], dtype=float)

        in_range = np.isfinite(u_candidates) & (u_candidates > u_min) & (u_candidates < u_max)
        if not np.any(in_range):
            return np.array([], dtype=float)

        return self._deduplicate_with_min_spacing(u_candidates[in_range], self._MIN_U_EDGE_SPACING)

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
        s_cdf = np.clip(s_cdf, 0.0, 1.0)
        s_survival = np.clip(s_survival, 0.0, 1.0)

        u2_vals = np.empty_like(s_cdf, dtype=float)
        lower_mask = s_cdf <= 0.5

        if np.any(lower_mask):
            lprobs = s_cdf[lower_mask]
            u2_vals[lower_mask] = beta_from_pf(lprobs, tail="lower")

        if np.any(~lower_mask):
            uprobs = s_survival[~lower_mask]
            u2_vals[~lower_mask] = beta_from_pf(uprobs, tail="upper")

        return u2_vals

    def _adaptive_probe_mass(
        self,
        u1_left: float,
        u1_right: float,
        u2_left: float,
        u2_right: float,
        split_factor: int,
    ) -> float:
        """Estimate cell failure mass by splitting and classifying subcell centers."""
        u1_sub_edges = np.linspace(u1_left, u1_right, split_factor + 1)
        u2_sub_edges = np.linspace(u2_left, u2_right, split_factor + 1)

        cdf_u1_sub, surv_u1_sub = self._normal_probabilities(u1_sub_edges)
        cdf_u2_sub, surv_u2_sub = self._normal_probabilities(u2_sub_edges)
        sub_prob_u1 = self._u_interval_probabilities(u1_sub_edges, cdf_u1_sub, surv_u1_sub)
        sub_prob_u2 = self._u_interval_probabilities(u2_sub_edges, cdf_u2_sub, surv_u2_sub)
        sub_weights = sub_prob_u1[:, None] * sub_prob_u2[None, :]
        u1_sub_centers = 0.5 * (u1_sub_edges[1:] + u1_sub_edges[:-1])
        u2_sub_centers = 0.5 * (u2_sub_edges[1:] + u2_sub_edges[:-1])
        u1_sub_mesh, u2_sub_mesh = np.meshgrid(u1_sub_centers, u2_sub_centers, indexing="ij")
        z_sub = self.evaluate_limit_state(u1_sub_mesh, u2_sub_mesh)
        return float(sub_weights[z_sub <= 0.0].sum())

    def _cell_failure_bounds(
        self,
        u1_left: float,
        u1_right: float,
        u2_left: float,
        u2_right: float,
    ) -> tuple[float, float]:
        """Return z_min/z_max bounds for a single cell."""
        u1_edges = np.array([u1_left, u1_right], dtype=float)
        u2_edges = np.array([u2_left, u2_right], dtype=float)

        u1_edges_cdf, u1_edges_survival = self._normal_probabilities(u1_edges)
        u2_edges_cdf, u2_edges_survival = self._normal_probabilities(u2_edges)
        r_edges = self._map_u_to_distribution(
            u1_edges,
            self.r_distribution,
            u_values_cdf=u1_edges_cdf,
            u_values_survival=u1_edges_survival,
        )
        s_edges = self._map_u_to_distribution(
            u2_edges,
            self.s_distribution,
            u_values_cdf=u2_edges_cdf,
            u_values_survival=u2_edges_survival,
        )
        z_min = float(r_edges[0] - s_edges[1])
        z_max = float(r_edges[1] - s_edges[0])
        return z_min, z_max

    def _build_adaptive_cell(
        self,
        u1_left: float,
        u1_right: float,
        u2_left: float,
        u2_right: float,
        weight: float,
        depth: int,
    ) -> _AdaptiveCell:
        """Construct an adaptive cell with center estimate and local error probe."""
        center_u1 = 0.5 * (u1_left + u1_right)
        center_u2 = 0.5 * (u2_left + u2_right)
        z_min, z_max = self._cell_failure_bounds(u1_left, u1_right, u2_left, u2_right)

        if z_max <= 0.0:
            return _AdaptiveCell(
                u1_left=u1_left,
                u1_right=u1_right,
                u2_left=u2_left,
                u2_right=u2_right,
                weight=float(weight),
                depth=depth,
                center_u1=center_u1,
                center_u2=center_u2,
                mass_est=float(weight),
                error_est=0.0,
                fail_estimate=True,
                refinable=False,
            )
        if z_min >= 0.0:
            return _AdaptiveCell(
                u1_left=u1_left,
                u1_right=u1_right,
                u2_left=u2_left,
                u2_right=u2_right,
                weight=float(weight),
                depth=depth,
                center_u1=center_u1,
                center_u2=center_u2,
                mass_est=0.0,
                error_est=0.0,
                fail_estimate=False,
                refinable=False,
            )

        z_center = float(
            self.evaluate_limit_state(
                np.array([center_u1], dtype=float),
                np.array([center_u2], dtype=float),
            )[0]
        )
        fail_estimate = z_center <= 0.0
        center_mass = float(weight) if fail_estimate else 0.0
        probe_mass = self._adaptive_probe_mass(
            u1_left=u1_left,
            u1_right=u1_right,
            u2_left=u2_left,
            u2_right=u2_right,
            split_factor=self.config.adaptive_probe_factor,
        )
        # Use probe mass as the mixed-cell estimate; center mass remains the
        # lower-order reference for the local error proxy.
        mass_est = float(probe_mass)
        error_est = float(abs(probe_mass - center_mass))
        if error_est == 0.0:
            # Probe and center can alias on thin mixed-cell slivers (both 0 or both
            # full mass). Keep a conservative local uncertainty so adaptive splitting
            # continues until corner bounds isolate the boundary.
            error_est = float(weight)
        refinable = depth < self.config.adaptive_max_depth and error_est > 0.0
        return _AdaptiveCell(
            u1_left=u1_left,
            u1_right=u1_right,
            u2_left=u2_left,
            u2_right=u2_right,
            weight=float(weight),
            depth=depth,
            center_u1=center_u1,
            center_u2=center_u2,
            mass_est=mass_est,
            error_est=error_est,
            fail_estimate=fail_estimate,
            refinable=refinable,
        )

    def _split_adaptive_cell(self, cell: _AdaptiveCell) -> list[_AdaptiveCell]:
        """Split an adaptive cell and evaluate child estimates."""
        split_factor = self.config.adaptive_split_factor
        u1_edges = np.linspace(cell.u1_left, cell.u1_right, split_factor + 1)
        u2_edges = np.linspace(cell.u2_left, cell.u2_right, split_factor + 1)
        u1_edges_cdf, u1_edges_survival = self._normal_probabilities(u1_edges)
        u2_edges_cdf, u2_edges_survival = self._normal_probabilities(u2_edges)
        u1_probs = self._u_interval_probabilities(u1_edges, u1_edges_cdf, u1_edges_survival)
        u2_probs = self._u_interval_probabilities(u2_edges, u2_edges_cdf, u2_edges_survival)

        children: list[_AdaptiveCell] = []
        for i in range(split_factor):
            for j in range(split_factor):
                child_weight = float(u1_probs[i] * u2_probs[j])
                if child_weight <= 0.0:
                    continue
                children.append(
                    self._build_adaptive_cell(
                        u1_left=float(u1_edges[i]),
                        u1_right=float(u1_edges[i + 1]),
                        u2_left=float(u2_edges[j]),
                        u2_right=float(u2_edges[j + 1]),
                        weight=child_weight,
                        depth=cell.depth + 1,
                    )
                )
        return children

    def _adaptive_leaf_cells(
        self,
        grid: _DistributionGridData,
    ) -> tuple[list[_AdaptiveCell], bool, int, float]:
        """Refine mixed cells adaptively and return final leaf cells with diagnostics."""
        cfg = self.config
        active_cells: dict[int, _AdaptiveCell] = {}
        heap: list[tuple[float, int]] = []
        next_id = 0
        for i, j in grid.mixed_indices:
            weight = float(grid.cell_weights[i, j])
            if weight <= 0.0:
                continue
            cell = self._build_adaptive_cell(
                u1_left=float(grid.u1_edges[i]),
                u1_right=float(grid.u1_edges[i + 1]),
                u2_left=float(grid.u2_edges[j]),
                u2_right=float(grid.u2_edges[j + 1]),
                weight=weight,
                depth=0,
            )
            active_cells[next_id] = cell
            if cell.refinable:
                heapq.heappush(heap, (-cell.error_est, next_id))
            next_id += 1

        coarse_pf = float(grid.cell_weights[grid.fail_mask].sum())
        mixed_pf = float(sum(cell.mass_est for cell in active_cells.values()))
        remaining_error = max(0.0, float(sum(cell.error_est for cell in active_cells.values())))
        iterations = 0
        prev_pf: float | None = None
        converged = False

        while True:
            pf_est = coarse_pf + mixed_pf
            denom = max(pf_est, cfg.adaptive_pf_floor)
            target_abs = (float(np.exp(cfg.adaptive_logpf_tol)) - 1.0) * denom
            beta_stable = True
            if prev_pf is not None:
                beta_prev = float(beta_from_pf(max(prev_pf, cfg.adaptive_pf_floor), tail="upper"))
                beta_curr = float(beta_from_pf(denom, tail="upper"))
                beta_stable = abs(beta_curr - beta_prev) <= cfg.adaptive_beta_tol

            if remaining_error <= target_abs and beta_stable:
                converged = True
                break

            if not heap:
                break

            _, cell_id = heapq.heappop(heap)
            cell = active_cells.get(cell_id)
            if cell is None or not cell.refinable:
                continue

            prev_pf = pf_est
            del active_cells[cell_id]
            children = self._split_adaptive_cell(cell)
            mixed_pf += -cell.mass_est + float(sum(child.mass_est for child in children))
            remaining_error = max(
                0.0,
                remaining_error + (-cell.error_est + float(sum(child.error_est for child in children))),
            )
            for child in children:
                active_cells[next_id] = child
                if child.refinable:
                    heapq.heappush(heap, (-child.error_est, next_id))
                next_id += 1
            iterations += 1

        return list(active_cells.values()), converged, iterations, remaining_error

    def _integrate_distributions_adaptive(self, grid: _DistributionGridData) -> FailureSamples:
        """Integrate failure mass with adaptive mixed-cell refinement."""
        cfg = self.config

        w_fail_coarse = grid.cell_weights[grid.fail_mask]
        u1_fail_coarse = grid.u1_center_grid[grid.fail_mask]
        u2_fail_coarse = grid.u2_center_grid[grid.fail_mask]

        adaptive_cells, converged, iterations, remaining_error = self._adaptive_leaf_cells(grid)

        w_fail_adaptive = [np.array([cell.mass_est], dtype=float) for cell in adaptive_cells if cell.mass_est > 0.0]
        u1_fail_adaptive = [np.array([cell.center_u1], dtype=float) for cell in adaptive_cells if cell.mass_est > 0.0]
        u2_fail_adaptive = [np.array([cell.center_u2], dtype=float) for cell in adaptive_cells if cell.mass_est > 0.0]

        weight_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        if w_fail_coarse.size > 0:
            weight_parts.append(w_fail_coarse)
            point_parts.append(np.column_stack([u1_fail_coarse, u2_fail_coarse]))
        if w_fail_adaptive:
            weight_parts.append(np.concatenate(w_fail_adaptive))
            point_parts.append(np.column_stack([np.concatenate(u1_fail_adaptive), np.concatenate(u2_fail_adaptive)]))

        if weight_parts:
            weights = np.concatenate(weight_parts)
            points = np.vstack(point_parts)
        else:
            weights = np.array([])
            points = np.empty((0, 2))

        final_pf = float(weights.sum()) if weights.size > 0 else 0.0
        denom = max(final_pf, cfg.adaptive_pf_floor)
        estimated_logpf_error = float(np.log1p(remaining_error / denom)) if remaining_error > 0.0 else 0.0
        max_depth_reached_cells = int(
            sum(1 for cell in adaptive_cells if cell.depth >= cfg.adaptive_max_depth and cell.error_est > 0.0)
        )
        refined_fail_cells = int(sum(1 for cell in adaptive_cells if cell.mass_est > 0.0))

        return FailureSamples(
            weights=weights,
            points=points,
            coarse_fail_cells=int(np.count_nonzero(grid.fail_mask)),
            refined_fail_cells=refined_fail_cells,
            mixed_cells=int(grid.mixed_indices.shape[0]),
            hazard_levels=None,
            adaptive_converged=converged,
            adaptive_iterations=iterations,
            adaptive_estimated_logpf_error=estimated_logpf_error,
            adaptive_remaining_pf_error=float(remaining_error),
            adaptive_max_depth_reached_cells=max_depth_reached_cells,
        )

    def _run_bounds_pass(
        self,
        bounds: tuple[float, float],
        previous_pass_pf: float | None,
    ) -> _BoundsPassResult:
        """Execute one full integration pass for fixed U-bounds."""
        cfg = self.config
        grid = self._compute_distribution_grid(u_min=bounds[0], u_max=bounds[1])
        samples = self._integrate_distributions_adaptive(grid)

        pf_est = float(samples.weights.sum()) if samples.weights.size > 0 else 0.0
        truncation_pf_error_bound = self._truncation_pf_error_bound(grid)
        adaptive_error = float(samples.adaptive_remaining_pf_error or 0.0)
        total_pf_error_bound = adaptive_error + truncation_pf_error_bound
        denom = max(pf_est, cfg.adaptive_pf_floor)
        target_abs = (float(np.exp(cfg.adaptive_logpf_tol)) - 1.0) * denom

        beta_stable = True
        if previous_pass_pf is not None:
            beta_prev = float(beta_from_pf(max(previous_pass_pf, cfg.adaptive_pf_floor), tail="upper"))
            beta_curr = float(beta_from_pf(denom, tail="upper"))
            beta_stable = abs(beta_curr - beta_prev) <= cfg.adaptive_beta_tol

        converged = total_pf_error_bound <= target_abs and beta_stable
        estimated_logpf_error = float(np.log1p(total_pf_error_bound / denom)) if total_pf_error_bound > 0.0 else 0.0

        samples.converged = converged
        samples.estimated_logpf_error = estimated_logpf_error
        samples.truncation_pf_error_bound = truncation_pf_error_bound
        samples.u_bounds_used = (float(grid.u_min), float(grid.u_max))

        return _BoundsPassResult(
            samples=samples,
            truncation_pf_error_bound=truncation_pf_error_bound,
            total_pf_error_bound=total_pf_error_bound,
            estimated_logpf_error=estimated_logpf_error,
            converged=converged,
            u_bounds_used=(float(grid.u_min), float(grid.u_max)),
        )

    # Distribution-based implementation
    def _integrate_distributions(self) -> FailureSamples:
        """Integrate probability mass with optional automatic bounds widening."""
        cfg = self.config
        auto_bounds = cfg.u_manual_bounds is None
        bounds = self._initial_u_bounds()
        previous_pass_pf: float | None = None
        latest_pass: _BoundsPassResult | None = None

        while True:
            pass_result = self._run_bounds_pass(bounds, previous_pass_pf)
            latest_pass = pass_result
            pf_est = float(pass_result.samples.weights.sum()) if pass_result.samples.weights.size > 0 else 0.0
            denom = max(pf_est, cfg.adaptive_pf_floor)
            target_abs = (float(np.exp(cfg.adaptive_logpf_tol)) - 1.0) * denom
            adaptive_error = float(pass_result.samples.adaptive_remaining_pf_error or 0.0)
            truncation_drives_failure = pass_result.truncation_pf_error_bound > max(0.0, target_abs - adaptive_error)
            beta_unstable_only = (pass_result.total_pf_error_bound <= target_abs) and (not pass_result.converged)

            if pass_result.converged:
                break
            if not auto_bounds:
                break
            if not (truncation_drives_failure or beta_unstable_only):
                break

            next_bounds = self._widen_bounds(bounds)
            if next_bounds is None:
                break
            previous_pass_pf = pf_est
            bounds = next_bounds

        if latest_pass is None:
            raise RuntimeError("Integration failed before producing any bounds pass.")
        return latest_pass.samples

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
        """
        w_fail = samples.weights
        U_fail = samples.points
        if w_fail.size == 0:
            # No failure cells detected during refinement.
            pf = 0.0
            beta_pf = float("inf")
            beta_star = float("nan")
            alpha_val = np.full((U_fail.shape[1],), np.nan)
        else:
            pf = float(w_fail.sum())
            beta_pf = float(beta_from_pf(pf, tail="upper"))
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
            adaptive_converged=samples.adaptive_converged,
            adaptive_iterations=samples.adaptive_iterations,
            adaptive_estimated_logpf_error=samples.adaptive_estimated_logpf_error,
            adaptive_remaining_pf_error=samples.adaptive_remaining_pf_error,
            adaptive_max_depth_reached_cells=samples.adaptive_max_depth_reached_cells,
            converged=samples.converged,
            estimated_logpf_error=samples.estimated_logpf_error,
            truncation_pf_error_bound=samples.truncation_pf_error_bound,
            u_bounds_used=samples.u_bounds_used,
        )
