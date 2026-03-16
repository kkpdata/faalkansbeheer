from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import openturns as ot
from scipy.optimize import minimize_scalar

from ..common.prob import beta_from_pf, pf_from_beta
from .config import IntegrationConfig
from .curve_distributions import FragilityDerivedDistribution, HazardDerivedDistribution
from .results import FailureSamples, IntegrationResult


@dataclass
class _IntegrationAxisGridData:
    u_min: float
    u_max: float
    u2_edges: np.ndarray
    u2_centers: np.ndarray
    interval_probs: np.ndarray
    include_mask: np.ndarray
    captured_mass: float
    target_mass: float
    u_s_max: float | None


@dataclass
class _AdaptiveInterval:
    u_left: float
    u_right: float
    prob_weight: float
    depth: int
    center_u2: float
    center_s: float
    center_u1: float
    mass_est: float
    error_est: float
    refinable: bool


@dataclass
class _BoundsPassResult:
    samples: FailureSamples
    truncation_pf_error_bound: float
    total_pf_error_bound: float
    estimated_logpf_error: float
    target_pf_error_bound: float
    converged: bool
    u_bounds_used: tuple[float, float]


class ReliabilityIntegrator:
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
        return self._r_distribution

    @property
    def s_distribution(self) -> ot.Distribution:
        return self._s_distribution

    @property
    def std_normal(self) -> ot.Normal:
        return self.config.std_normal

    def standard_normal_cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        values = pf_from_beta(x, tail="lower")
        if np.isscalar(x):
            return float(values)
        return values

    def evaluate_limit_state(self, u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
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
            neg_part = 0.5 - cdf_left[cross_mask]
            pos_part = 0.5 - surv_right[cross_mask]
            probs[cross_mask] = neg_part + pos_part

        return probs

    def _distribution_probabilities_at_levels(
        self,
        dist: ot.Distribution,
        levels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(levels, dtype=float).reshape(-1)
        cdf = np.array(dist.computeCDF(arr[:, np.newaxis])).reshape(-1)
        survival = np.array(dist.computeSurvivalFunction(arr[:, np.newaxis])).reshape(-1)
        cdf = np.clip(cdf, 0.0, 1.0)
        survival = np.clip(survival, 0.0, 1.0)
        return cdf.reshape(np.shape(levels)), survival.reshape(np.shape(levels))

    def run(self) -> IntegrationResult:
        samples = self.integrate_failure_samples()
        return self._postprocess_failure_samples(samples)

    def integrate_failure_samples(self) -> FailureSamples:
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
        lower, upper = float(bounds[0]), float(bounds[1])
        widened_lower = max(-self._AUTO_BOUNDS_MAX_ABS, lower - self._AUTO_BOUNDS_WIDEN_STEP)
        widened_upper = min(self._AUTO_BOUNDS_MAX_ABS, upper + self._AUTO_BOUNDS_WIDEN_STEP)
        if widened_lower == lower and widened_upper == upper:
            return None
        return widened_lower, widened_upper

    def _truncation_pf_error_bound(self, grid: _IntegrationAxisGridData) -> float:
        return max(0.0, float(grid.target_mass - grid.captured_mass))

    def _relative_pf_error_target(self, pf_est: float) -> float:
        denom = max(float(pf_est), self.config.adaptive_pf_floor)
        return (float(np.exp(self.config.adaptive_logpf_tol)) - 1.0) * denom

    def _combined_pf_error_target(self, pf_est: float) -> float:
        rel_target = self._relative_pf_error_target(pf_est)
        abs_tol = self.config.adaptive_abs_pf_tol
        if abs_tol is None:
            return rel_target
        return max(rel_target, float(abs_tol))

    def _convergence_reason(
        self,
        *,
        total_pf_error_bound: float,
        truncation_pf_error_bound: float,
        adaptive_pf_error_bound: float,
        target_pf_error_bound: float,
        relative_target_pf_error_bound: float,
        converged: bool,
    ) -> str:
        if converged:
            abs_tol = self.config.adaptive_abs_pf_tol
            if (
                abs_tol is not None
                and target_pf_error_bound > relative_target_pf_error_bound
                and total_pf_error_bound > relative_target_pf_error_bound
            ):
                return "converged_absolute_tol"
            return "converged_relative_tol"

        beta_unstable_only = total_pf_error_bound <= target_pf_error_bound
        if beta_unstable_only:
            return "beta_unstable"

        truncation_limited = truncation_pf_error_bound > max(0.0, target_pf_error_bound - adaptive_pf_error_bound)
        if truncation_limited:
            return "truncation_limited"
        return "adaptive_limited"

    @staticmethod
    def _merge_level_candidates(*level_groups: np.ndarray | list[float] | None) -> np.ndarray:
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
        levels: list[float] = []

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
    ) -> _IntegrationAxisGridData:
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
        curve_levels = self._merge_level_candidates(
            cfg.hazard_curve.hazard_levels if cfg.hazard_curve is not None else None,
            cfg.fragility_curve.hazard_levels if cfg.fragility_curve is not None else None,
        )

        s_x_min, s_x_max = self._distribution_value_bounds(self.s_distribution, u_min=u_min, u_max=u_max)
        curve_injected = self._clip_levels_to_range(curve_levels, x_min=s_x_min, x_max=s_x_max)
        s_step_levels = self._distribution_step_levels(self.s_distribution, x_min=s_x_min, x_max=s_x_max)
        r_step_levels = self._distribution_step_levels(self.r_distribution, x_min=s_x_min, x_max=s_x_max)
        mandatory_levels = self._merge_level_candidates(curve_injected, s_step_levels, r_step_levels)

        u2_edges = self._build_axis_edges(
            dist=self.s_distribution,
            curve_levels=mandatory_levels,
            u_min=u_min,
            u_max=u_max,
        )

        u_s_max = None
        target_mass = 1.0
        if cfg.max_solicitation_level is not None:
            prob = float(self.s_distribution.computeCDF(cfg.max_solicitation_level))
            prob = float(np.clip(prob, 0.0, 1.0))
            target_mass = prob
            u_s_max = float(beta_from_pf(prob, tail="lower"))
            if u_min < u_s_max < u_max:
                u2_edges = np.sort(np.unique(np.r_[u2_edges, u_s_max]))

        u2_centers = 0.5 * (u2_edges[1:] + u2_edges[:-1])
        u2_edges_cdf, u2_edges_survival = self._normal_probabilities(u2_edges)
        u2_probs = self._u_interval_probabilities(u2_edges, u2_edges_cdf, u2_edges_survival)
        if np.any(~np.isfinite(u2_probs)) or np.any(u2_probs < 0.0):
            raise RuntimeError("Invalid U-grid configuration produced invalid probability mass.")

        include_mask = np.ones(u2_edges.size - 1, dtype=bool)
        if u_s_max is not None:
            if u_s_max <= u2_edges[0]:
                include_mask = np.zeros(u2_edges.size - 1, dtype=bool)
            elif u_s_max < u2_edges[-1]:
                include_mask = u2_edges[1:] <= u_s_max
            u2_probs = np.where(include_mask, u2_probs, 0.0)

        captured_mass = float(u2_probs.sum())
        if not np.isfinite(captured_mass) or captured_mass < 0.0:
            raise RuntimeError("Invalid U-grid configuration produced invalid probability mass.")

        return _IntegrationAxisGridData(
            u_min=u_min,
            u_max=u_max,
            u2_edges=u2_edges,
            u2_centers=u2_centers,
            interval_probs=u2_probs,
            include_mask=include_mask,
            captured_mass=captured_mass,
            target_mass=float(target_mass),
            u_s_max=u_s_max,
        )

    def _build_axis_edges(
        self,
        dist: ot.Distribution,
        curve_levels: np.ndarray | None,
        u_min: float,
        u_max: float,
    ) -> np.ndarray:
        cfg = self.config
        mandatory = np.array([u_min, u_max], dtype=float)
        mapped_knots = self._map_curve_levels_to_u_edges(dist=dist, levels=curve_levels, u_min=u_min, u_max=u_max)
        if mapped_knots.size > 0:
            mandatory = np.r_[mandatory, mapped_knots]
        mandatory_edges = self._deduplicate_with_min_spacing(mandatory, self._MIN_U_EDGE_SPACING)

        if mandatory_edges.size >= cfg.coarse_points:
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
        if levels is None:
            return np.array([], dtype=float)

        level_arr = np.asarray(levels, dtype=float).reshape(-1)
        if level_arr.size == 0:
            return np.array([], dtype=float)

        level_arr = level_arr[np.isfinite(level_arr)]
        if level_arr.size == 0:
            return np.array([], dtype=float)
        level_arr = self._deduplicate_with_min_spacing(level_arr, self._CURVE_LEVEL_MIN_SPACING_X)
        if level_arr.size == 0:
            return np.array([], dtype=float)

        cdf, survival = self._distribution_probabilities_at_levels(dist, level_arr)
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

    def _failure_cdf_and_u_for_r_equals_s(
        self,
        s_values: np.ndarray,
        *,
        include_u1_equivalent: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate failure CDF and (optionally) equivalent U1 at physical levels."""
        arr = np.asarray(s_values, dtype=float)
        cdf, survival = self._distribution_probabilities_at_levels(self.r_distribution, arr)
        cdf_arr = np.asarray(cdf, dtype=float)
        survival_arr = np.asarray(survival, dtype=float)
        fail_cdf = np.clip(np.where(cdf_arr <= 0.5, cdf_arr, 1.0 - survival_arr), 0.0, 1.0)

        if not include_u1_equivalent:
            return fail_cdf, None

        flat_cdf = cdf_arr.reshape(-1)
        flat_survival = survival_arr.reshape(-1)
        u1_flat = np.empty_like(flat_cdf, dtype=float)
        lower_mask = flat_cdf <= 0.5
        if np.any(lower_mask):
            u1_flat[lower_mask] = beta_from_pf(flat_cdf[lower_mask], tail="lower")
        if np.any(~lower_mask):
            u1_flat[~lower_mask] = beta_from_pf(flat_survival[~lower_mask], tail="upper")
        return fail_cdf, u1_flat.reshape(arr.shape)

    def _failure_cdf_at_s(self, s_values: np.ndarray) -> np.ndarray:
        fail_cdf, _ = self._failure_cdf_and_u_for_r_equals_s(s_values, include_u1_equivalent=False)
        return fail_cdf

    def _u_for_r_equals_s(self, s_values: np.ndarray) -> np.ndarray:
        _, u1_vals = self._failure_cdf_and_u_for_r_equals_s(s_values, include_u1_equivalent=True)
        if u1_vals is None:
            raise RuntimeError("Internal error: expected u1 equivalent values.")
        return u1_vals

    def _evaluate_u2_nodes(
        self,
        u2_values: np.ndarray,
        *,
        include_u1_equivalent: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Map U2 nodes to physical S and evaluate R-based failure metrics."""
        u2_arr = np.asarray(u2_values, dtype=float)
        cdf_vals, survival_vals = self._normal_probabilities(u2_arr, compute_cdf=True, compute_survival=True)
        s_vals = self._map_u_to_distribution(
            u2_arr,
            self.s_distribution,
            u_values_cdf=cdf_vals,
            u_values_survival=survival_vals,
        )
        fail_cdf, u1_vals = self._failure_cdf_and_u_for_r_equals_s(
            s_vals,
            include_u1_equivalent=include_u1_equivalent,
        )
        return s_vals, fail_cdf, u1_vals

    def _adaptive_probe_masses(
        self,
        u_left: np.ndarray,
        u_right: np.ndarray,
        split_factor: int,
    ) -> np.ndarray:
        left = np.asarray(u_left, dtype=float).reshape(-1)
        right = np.asarray(u_right, dtype=float).reshape(-1)
        if left.size != right.size:
            raise ValueError("u_left and u_right must have the same size.")
        if left.size == 0:
            return np.array([], dtype=float)

        fractions = np.linspace(0.0, 1.0, split_factor + 1)
        u_sub_edges = left[:, None] + (right - left)[:, None] * fractions[None, :]
        cdf_sub, survival_sub = self._normal_probabilities(u_sub_edges, compute_cdf=True, compute_survival=True)
        sub_probs = self._u_interval_probabilities(u_sub_edges, cdf_sub, survival_sub)
        u_sub_centers = 0.5 * (u_sub_edges[:, 1:] + u_sub_edges[:, :-1])
        _, f_sub, _ = self._evaluate_u2_nodes(u_sub_centers, include_u1_equivalent=False)
        return np.sum(sub_probs * np.asarray(f_sub, dtype=float), axis=1)

    def _adaptive_probe_mass(
        self,
        u_left: float,
        u_right: float,
        split_factor: int,
    ) -> float:
        masses = self._adaptive_probe_masses(
            np.array([u_left], dtype=float),
            np.array([u_right], dtype=float),
            split_factor=split_factor,
        )
        if masses.size == 0:
            return 0.0
        return float(masses[0])

    def _build_adaptive_intervals_batch(
        self,
        u_left: np.ndarray,
        u_right: np.ndarray,
        prob_weight: np.ndarray,
        depth: int,
    ) -> list[_AdaptiveInterval]:
        left = np.asarray(u_left, dtype=float).reshape(-1)
        right = np.asarray(u_right, dtype=float).reshape(-1)
        weight = np.asarray(prob_weight, dtype=float).reshape(-1)
        if left.size != right.size or left.size != weight.size:
            raise ValueError("u_left, u_right, and prob_weight must have identical sizes.")
        if left.size == 0:
            return []

        centers = 0.5 * (left + right)
        center_s, center_f, center_u1 = self._evaluate_u2_nodes(centers, include_u1_equivalent=True)
        if center_u1 is None:
            raise RuntimeError("Internal error: center U1 values were not computed.")
        center_mass = weight * np.asarray(center_f, dtype=float).reshape(-1)
        probe_mass = self._adaptive_probe_masses(left, right, split_factor=self.config.adaptive_probe_factor)
        error_est = np.abs(probe_mass - center_mass)
        next_depth = int(depth)
        can_refine = next_depth < self.config.adaptive_max_depth

        intervals: list[_AdaptiveInterval] = []
        center_s_flat = np.asarray(center_s, dtype=float).reshape(-1)
        center_u1_flat = np.asarray(center_u1, dtype=float).reshape(-1)
        for i in range(left.size):
            interval_error = float(error_est[i])
            intervals.append(
                _AdaptiveInterval(
                    u_left=float(left[i]),
                    u_right=float(right[i]),
                    prob_weight=float(weight[i]),
                    depth=next_depth,
                    center_u2=float(centers[i]),
                    center_s=float(center_s_flat[i]),
                    center_u1=float(center_u1_flat[i]),
                    mass_est=float(probe_mass[i]),
                    error_est=interval_error,
                    refinable=can_refine and interval_error > 0.0,
                )
            )
        return intervals

    def _build_adaptive_interval(
        self,
        u_left: float,
        u_right: float,
        prob_weight: float,
        depth: int,
    ) -> _AdaptiveInterval:
        intervals = self._build_adaptive_intervals_batch(
            np.array([u_left], dtype=float),
            np.array([u_right], dtype=float),
            np.array([prob_weight], dtype=float),
            depth=int(depth),
        )
        if not intervals:
            raise RuntimeError("Internal error: expected one adaptive interval.")
        return intervals[0]

    def _split_adaptive_interval(self, interval: _AdaptiveInterval) -> list[_AdaptiveInterval]:
        split_factor = self.config.adaptive_split_factor
        u_edges = np.linspace(interval.u_left, interval.u_right, split_factor + 1)
        cdf_edges, survival_edges = self._normal_probabilities(u_edges, compute_cdf=True, compute_survival=True)
        sub_probs = self._u_interval_probabilities(u_edges, cdf_edges, survival_edges)
        child_mask = sub_probs > 0.0
        if not np.any(child_mask):
            return []
        return self._build_adaptive_intervals_batch(
            u_edges[:-1][child_mask],
            u_edges[1:][child_mask],
            sub_probs[child_mask],
            depth=interval.depth + 1,
        )

    def _adaptive_leaf_intervals(
        self,
        grid: _IntegrationAxisGridData,
    ) -> tuple[list[_AdaptiveInterval], bool, int, float, int, int]:
        cfg = self.config
        active: dict[int, _AdaptiveInterval] = {}
        heap: list[tuple[float, int]] = []
        next_id = 0
        coarse_fail_intervals = 0
        mixed_intervals = 0

        positive_mask = np.asarray(grid.interval_probs, dtype=float) > 0.0
        positive_indices = np.nonzero(positive_mask)[0]
        initial_intervals = self._build_adaptive_intervals_batch(
            grid.u2_edges[positive_indices],
            grid.u2_edges[positive_indices + 1],
            np.asarray(grid.interval_probs, dtype=float)[positive_indices],
            depth=0,
        )

        for interval in initial_intervals:
            if interval.mass_est > 0.0:
                coarse_fail_intervals += 1
            if interval.refinable:
                mixed_intervals += 1
            active[next_id] = interval
            if interval.refinable:
                heapq.heappush(heap, (-interval.error_est, next_id))
            next_id += 1

        pf_est = float(sum(interval.mass_est for interval in active.values()))
        remaining_error = max(0.0, float(sum(interval.error_est for interval in active.values())))
        iterations = 0
        prev_pf: float | None = None
        converged = False

        while True:
            target_abs = self._combined_pf_error_target(pf_est)
            beta_stable = True
            if prev_pf is not None:
                beta_prev = float(beta_from_pf(max(prev_pf, cfg.adaptive_pf_floor), tail="upper"))
                beta_curr = float(beta_from_pf(max(pf_est, cfg.adaptive_pf_floor), tail="upper"))
                beta_stable = abs(beta_curr - beta_prev) <= cfg.adaptive_beta_tol

            if remaining_error <= target_abs and beta_stable:
                converged = True
                break

            if not heap:
                break

            _, interval_id = heapq.heappop(heap)
            interval = active.get(interval_id)
            if interval is None or not interval.refinable:
                continue

            prev_pf = pf_est
            del active[interval_id]
            children = self._split_adaptive_interval(interval)
            child_mass = float(sum(child.mass_est for child in children))
            child_error = float(sum(child.error_est for child in children))
            pf_est = max(0.0, pf_est - interval.mass_est + child_mass)
            remaining_error = max(0.0, remaining_error - interval.error_est + child_error)

            for child in children:
                active[next_id] = child
                if child.refinable:
                    heapq.heappush(heap, (-child.error_est, next_id))
                next_id += 1

            iterations += 1

        return list(active.values()), converged, iterations, remaining_error, coarse_fail_intervals, mixed_intervals

    def _integrate_distributions_adaptive(self, grid: _IntegrationAxisGridData) -> FailureSamples:
        cfg = self.config
        intervals, converged, iterations, remaining_error, coarse_fail_intervals, mixed_intervals = (
            self._adaptive_leaf_intervals(grid)
        )

        positive = [interval for interval in intervals if interval.mass_est > 0.0]
        if positive:
            weights = np.array([interval.mass_est for interval in positive], dtype=float)
            u1 = np.array([interval.center_u1 for interval in positive], dtype=float)
            u2 = np.array([interval.center_u2 for interval in positive], dtype=float)
            points = np.column_stack([u1, u2])
        else:
            weights = np.array([], dtype=float)
            points = np.empty((0, 2))

        final_pf = float(weights.sum()) if weights.size > 0 else 0.0
        denom = max(final_pf, cfg.adaptive_pf_floor)
        estimated_logpf_error = float(np.log1p(remaining_error / denom)) if remaining_error > 0.0 else 0.0
        max_depth_reached_intervals = int(
            sum(1 for interval in intervals if interval.depth >= cfg.adaptive_max_depth and interval.error_est > 0.0)
        )
        refined_fail_intervals = int(len(positive))

        return FailureSamples(
            weights=weights,
            points=points,
            coarse_fail_cells=coarse_fail_intervals,
            refined_fail_cells=refined_fail_intervals,
            mixed_cells=mixed_intervals,
            hazard_levels=None,
            adaptive_converged=converged,
            adaptive_iterations=iterations,
            adaptive_estimated_logpf_error=estimated_logpf_error,
            adaptive_remaining_pf_error=float(remaining_error),
            adaptive_max_depth_reached_cells=max_depth_reached_intervals,
        )

    def _run_bounds_pass(
        self,
        bounds: tuple[float, float],
        previous_pass_pf: float | None,
    ) -> _BoundsPassResult:
        cfg = self.config
        grid = self._compute_distribution_grid(u_min=bounds[0], u_max=bounds[1])
        samples = self._integrate_distributions_adaptive(grid)

        pf_est = float(samples.weights.sum()) if samples.weights.size > 0 else 0.0
        truncation_pf_error_bound = self._truncation_pf_error_bound(grid)
        adaptive_error = float(samples.adaptive_remaining_pf_error or 0.0)
        total_pf_error_bound = adaptive_error + truncation_pf_error_bound
        denom = max(pf_est, cfg.adaptive_pf_floor)
        relative_target = self._relative_pf_error_target(pf_est)
        target_abs = self._combined_pf_error_target(pf_est)

        beta_stable = True
        if previous_pass_pf is not None:
            beta_prev = float(beta_from_pf(max(previous_pass_pf, cfg.adaptive_pf_floor), tail="upper"))
            beta_curr = float(beta_from_pf(denom, tail="upper"))
            beta_stable = abs(beta_curr - beta_prev) <= cfg.adaptive_beta_tol

        converged = total_pf_error_bound <= target_abs and beta_stable
        estimated_logpf_error = float(np.log1p(total_pf_error_bound / denom)) if total_pf_error_bound > 0.0 else 0.0

        samples.converged = converged
        samples.convergence_reason = self._convergence_reason(
            total_pf_error_bound=total_pf_error_bound,
            truncation_pf_error_bound=truncation_pf_error_bound,
            adaptive_pf_error_bound=adaptive_error,
            target_pf_error_bound=target_abs,
            relative_target_pf_error_bound=relative_target,
            converged=converged,
        )
        samples.estimated_logpf_error = estimated_logpf_error
        samples.truncation_pf_error_bound = truncation_pf_error_bound
        samples.u_bounds_used = (float(grid.u_min), float(grid.u_max))

        return _BoundsPassResult(
            samples=samples,
            truncation_pf_error_bound=truncation_pf_error_bound,
            total_pf_error_bound=total_pf_error_bound,
            estimated_logpf_error=estimated_logpf_error,
            target_pf_error_bound=target_abs,
            converged=converged,
            u_bounds_used=(float(grid.u_min), float(grid.u_max)),
        )

    def _integrate_distributions(self) -> FailureSamples:
        cfg = self.config
        auto_bounds = cfg.u_manual_bounds is None
        bounds = self._initial_u_bounds()
        previous_pass_pf: float | None = None
        latest_pass: _BoundsPassResult | None = None
        final_reason: str | None = None

        while True:
            pass_result = self._run_bounds_pass(bounds, previous_pass_pf)
            latest_pass = pass_result
            pf_est = float(pass_result.samples.weights.sum()) if pass_result.samples.weights.size > 0 else 0.0
            target_abs = pass_result.target_pf_error_bound
            adaptive_error = float(pass_result.samples.adaptive_remaining_pf_error or 0.0)
            truncation_drives_failure = pass_result.truncation_pf_error_bound > max(0.0, target_abs - adaptive_error)
            beta_unstable_only = (pass_result.total_pf_error_bound <= target_abs) and (not pass_result.converged)

            if pass_result.converged:
                final_reason = pass_result.samples.convergence_reason
                break
            if not auto_bounds:
                final_reason = pass_result.samples.convergence_reason
                break
            if not (truncation_drives_failure or beta_unstable_only):
                final_reason = pass_result.samples.convergence_reason
                break

            next_bounds = self._widen_bounds(bounds)
            if next_bounds is None:
                final_reason = pass_result.samples.convergence_reason
                break
            previous_pass_pf = pf_est
            bounds = next_bounds

        if latest_pass is None:
            raise RuntimeError("Integration failed before producing any bounds pass.")
        if final_reason is not None:
            latest_pass.samples.convergence_reason = final_reason
        return latest_pass.samples

    def _initialize_distributions(self) -> None:
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

    def _limit_state_curve(self, u_values: np.ndarray) -> np.ndarray:
        """Map ``u1`` values to ``u2`` values on the transformed ``g=0`` curve.

        For each ``u1``:
        1. Map to physical resistance ``r`` via the resistance marginal.
        2. Evaluate solicitation CDF/SF at ``r``.
        3. Convert that probability back to a numerically stable ``u2``.
        """
        u_values = np.asarray(u_values, dtype=float)
        cdf_u, survival_u = self._normal_probabilities(u_values, compute_cdf=True, compute_survival=True)
        r_vals = self._map_u_to_distribution(
            u_values,
            self.r_distribution,
            u_values_cdf=cdf_u,
            u_values_survival=survival_u,
        )

        s_cdf, s_survival = self._distribution_probabilities_at_levels(self.s_distribution, r_vals)
        s_cdf = np.asarray(s_cdf, dtype=float).reshape(-1)
        s_survival = np.asarray(s_survival, dtype=float).reshape(-1)

        u2_vals = np.empty_like(s_cdf, dtype=float)
        lower_mask = s_cdf <= 0.5
        if np.any(lower_mask):
            u2_vals[lower_mask] = beta_from_pf(s_cdf[lower_mask], tail="lower")
        if np.any(~lower_mask):
            u2_vals[~lower_mask] = beta_from_pf(s_survival[~lower_mask], tail="upper")
        return u2_vals.reshape(u_values.shape)

    def _solve_design_point(
        self,
        bounds: tuple[float, float],
        u_s_max: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Find the closest point to origin on the transformed g=0 curve.

        Method:
        1. Build objective J(u1) = u1^2 + u2(u1)^2 using ``_limit_state_curve``.
        2. Enforce feasibility by penalizing non-finite points and optional
           solicitation cutoff (u2 > u_s_max).
        3. Seed from coarse scan, split feasible scan into contiguous intervals,
           and run bounded optimization per interval (plus local-minima brackets).
        4. Return ``beta_star = sqrt(min J)`` and ``alpha = -u*/beta_star``;
           if ``beta_star == 0`` then ``alpha = [nan, nan]``.
        """
        u_min, u_max = float(bounds[0]), float(bounds[1])
        if not (u_max > u_min):
            return float("nan"), np.array([np.nan, np.nan], dtype=float)

        # Large finite penalty keeps bounded optimizers away from invalid points
        # while still allowing recovery if the valid region is narrow.
        invalid_penalty = 1.0e50

        def evaluate_feasible(u1: float) -> tuple[float, float] | None:
            u2 = float(self._limit_state_curve(np.array([u1], dtype=float))[0])
            if not np.isfinite(u2):
                return None
            if u_s_max is not None and u2 > u_s_max + 1e-12:
                return None
            obj = float(u1 * u1 + u2 * u2)
            if not np.isfinite(obj):
                return None
            return obj, u2

        # Step 1: Objective on the implicit g=0 curve, J(u1)=u1^2+u2(u1)^2.
        def objective(u1: float) -> float:
            evaluated = evaluate_feasible(float(u1))
            if evaluated is None:
                return invalid_penalty
            return float(evaluated[0])

        # Step 2: Coarse global scan provides a robust seed and fallback.
        n_scan = max(257, int(4 * self.config.coarse_points + 1))
        u_scan = np.linspace(u_min, u_max, n_scan)
        u2_scan = self._limit_state_curve(u_scan)
        mask = np.isfinite(u2_scan)
        if u_s_max is not None:
            mask = mask & (u2_scan <= (u_s_max + 1e-12))
        if not np.any(mask):
            return float("nan"), np.array([np.nan, np.nan], dtype=float)

        valid_indices = np.nonzero(mask)[0]
        obj_scan = u_scan[mask] ** 2 + u2_scan[mask] ** 2
        local_best_idx = int(np.argmin(obj_scan))
        best_index = int(valid_indices[local_best_idx])
        best_u1 = float(u_scan[best_index])
        best_u2 = float(u2_scan[best_index])
        best_obj = float(obj_scan[local_best_idx])

        # Step 3: Split feasible scan into contiguous intervals and optimize each.
        # This is more robust than a single full-range bounded solve when J(u1)
        # is non-unimodal due to kinks/plateaus in transformed distributions.
        obj_full = np.full_like(u_scan, np.inf, dtype=float)
        obj_full[mask] = obj_scan

        feasible_segments: list[tuple[int, int]] = []
        seg_start = int(valid_indices[0])
        seg_end = seg_start
        for idx in valid_indices[1:]:
            idx_i = int(idx)
            if idx_i == seg_end + 1:
                seg_end = idx_i
            else:
                feasible_segments.append((seg_start, seg_end))
                seg_start = idx_i
                seg_end = idx_i
        feasible_segments.append((seg_start, seg_end))

        interval_candidates: list[tuple[float, float]] = []
        for seg_left_idx, seg_right_idx in feasible_segments:
            left = float(u_scan[seg_left_idx])
            right = float(u_scan[seg_right_idx])
            if right > left:
                interval_candidates.append((left, right))

            if seg_right_idx - seg_left_idx >= 2:
                for idx in range(seg_left_idx + 1, seg_right_idx):
                    if obj_full[idx] <= obj_full[idx - 1] and obj_full[idx] <= obj_full[idx + 1]:
                        local_left = float(u_scan[idx - 1])
                        local_right = float(u_scan[idx + 1])
                        if local_right > local_left:
                            interval_candidates.append((local_left, local_right))

        deduped_intervals: list[tuple[float, float]] = []
        seen_intervals: set[tuple[float, float]] = set()
        for left, right in interval_candidates:
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            if right <= left:
                continue
            key = (round(float(left), 12), round(float(right), 12))
            if key in seen_intervals:
                continue
            seen_intervals.add(key)
            deduped_intervals.append((float(left), float(right)))

        def update_best(u1: float) -> None:
            nonlocal best_obj, best_u1, best_u2
            evaluated = evaluate_feasible(float(u1))
            if evaluated is None:
                return
            obj_val, u2_val = evaluated
            if obj_val < best_obj:
                best_obj = float(obj_val)
                best_u1 = float(u1)
                best_u2 = float(u2_val)

        for left, right in deduped_intervals:
            # Boundary checks protect against boundary minima from truncation.
            update_best(left)
            update_best(right)
            try:
                interval_result = minimize_scalar(objective, bounds=(left, right), method="bounded")
                if interval_result.success:
                    update_best(float(interval_result.x))
            except Exception:
                pass

        if not np.isfinite(best_obj) or not np.isfinite(best_u1) or not np.isfinite(best_u2):
            return float("nan"), np.array([np.nan, np.nan], dtype=float)

        # Step 4: Convert minimum distance to FORM-style outputs.
        beta_star = float(np.sqrt(max(0.0, best_obj)))
        # Deterministic policy at origin: no divide-by-zero, alpha is undefined.
        if beta_star == 0.0:
            return beta_star, np.array([np.nan, np.nan], dtype=float)
        alpha = -np.array([best_u1, best_u2], dtype=float) / beta_star
        return beta_star, alpha

    def _postprocess_failure_samples(self, samples: FailureSamples) -> IntegrationResult:
        w_fail = samples.weights
        if w_fail.size == 0:
            pf = 0.0
            beta_pf = float("inf")
            beta_star = float("nan")
            alpha_val = np.array([np.nan, np.nan], dtype=float)
        else:
            pf = float(w_fail.sum())
            # beta_pf is derived from integrated Pf; beta_star/alpha come from
            # geometric closest-point search on the transformed g=0 curve.
            beta_pf = float(beta_from_pf(pf, tail="upper"))
            bounds = samples.u_bounds_used if samples.u_bounds_used is not None else self._initial_u_bounds()
            u_s_max = None
            if self.config.max_solicitation_level is not None:
                prob = float(self.s_distribution.computeCDF(self.config.max_solicitation_level))
                prob = float(np.clip(prob, 0.0, 1.0))
                u_s_max = float(beta_from_pf(prob, tail="lower"))
            beta_star, alpha_val = self._solve_design_point(bounds, u_s_max=u_s_max)

        hazard_level = None
        if self.config.hazard_curve is not None and np.isfinite(beta_star):
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
            convergence_reason=samples.convergence_reason,
            estimated_logpf_error=samples.estimated_logpf_error,
            truncation_pf_error_bound=samples.truncation_pf_error_bound,
            u_bounds_used=samples.u_bounds_used,
        )
