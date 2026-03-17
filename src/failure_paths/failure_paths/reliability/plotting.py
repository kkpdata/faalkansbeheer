from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .results import IntegrationResult


def _sorted_trace_arrays(result: IntegrationResult) -> dict[str, np.ndarray]:
    """Return integration trace arrays sorted by U2 center.

    Parameters
    ----------
    result : IntegrationResult
        Reliability integration result that may hold interval-level trace data.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of aligned trace arrays sorted by ``u_center``.

    Raises
    ------
    ValueError
        If diagnostics trace is missing or empty.
    """
    trace = result.failure_samples.integration_trace
    if trace is None:
        raise ValueError(
            "Integration diagnostics trace is unavailable. Run the integrator with collect_diagnostics_trace=True."
        )
    if trace.u_center.size == 0:
        raise ValueError("Integration diagnostics trace is empty.")

    order = np.argsort(np.asarray(trace.u_center, dtype=float))
    return {
        "u_left": np.asarray(trace.u_left, dtype=float)[order],
        "u_right": np.asarray(trace.u_right, dtype=float)[order],
        "u_center": np.asarray(trace.u_center, dtype=float)[order],
        "depth": np.asarray(trace.depth, dtype=int)[order],
        "prob_weight": np.asarray(trace.prob_weight, dtype=float)[order],
        "failure_cdf_center": np.asarray(trace.failure_cdf_center, dtype=float)[order],
        "local_pf_contribution": np.asarray(trace.local_pf_contribution, dtype=float)[order],
        "error_estimate": np.asarray(trace.error_estimate, dtype=float)[order],
        "center_u1": np.asarray(trace.center_u1, dtype=float)[order],
    }


def plot_integration_diagnostics_1d(
    result: IntegrationResult,
    axes: Axes | np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (7.5, 5.0),
) -> tuple[Figure, np.ndarray, dict[str, np.ndarray | float]]:
    """Render the limit-state signal in U-space with failure/no-failure shading.

    Parameters
    ----------
    result : IntegrationResult
        Reliability integration result containing an optional diagnostics trace.
    axes : Axes | np.ndarray | None
        Optional axis container with exactly one axis. When omitted, a new
        figure with one axis is created.
    figsize : tuple[float, float]
        Figure size used when ``axes`` is ``None``.

    Returns
    -------
    tuple[Figure, np.ndarray, dict[str, np.ndarray | float]]
        Figure handle, one-axis array, and diagnostics arrays used for plotting.

    Raises
    ------
    ValueError
        If diagnostics trace data is unavailable or ``axes`` does not contain
        exactly one axis.
    """
    data = _sorted_trace_arrays(result)

    u_left = data["u_left"]
    u_right = data["u_right"]
    u_center = data["u_center"]
    depth = data["depth"].astype(float)
    prob_weight = np.maximum(data["prob_weight"], 0.0)
    failure_cdf_center = data["failure_cdf_center"]
    local_pf = np.maximum(data["local_pf_contribution"], 0.0)
    error_estimate = np.maximum(data["error_estimate"], 0.0)
    center_u1_raw = data["center_u1"]
    center_u1 = np.where(np.isfinite(center_u1_raw), center_u1_raw, np.nan)

    if axes is None:
        fig, axis = plt.subplots(figsize=figsize)
        axes_arr = np.asarray([axis], dtype=object)
    else:
        axes_arr = np.asarray(axes, dtype=object).reshape(-1)
        if axes_arr.size != 1:
            raise ValueError("axes must contain exactly one Matplotlib axis.")
        fig = axes_arr[0].figure

    ax_limit = axes_arr[0]

    y_candidates = center_u1[np.isfinite(center_u1)]
    if y_candidates.size == 0:
        y_min, y_max = -5.0, 5.0
    else:
        y_low = float(np.min(y_candidates))
        y_high = float(np.max(y_candidates))
        if np.isclose(y_low, y_high):
            y_low -= 1.0
            y_high += 1.0
        margin = 0.12 * (y_high - y_low)
        y_min = y_low - margin
        y_max = y_high + margin

    x_left = float(np.min(u_left))
    x_right = float(np.max(u_right))
    ax_limit.set_xlim(x_left, x_right)
    ax_limit.set_ylim(y_min, y_max)

    # For full-width background shading, clip non-finite limit-state values:
    # -inf -> y_min (all no-failure), +inf -> y_max (all failure).
    boundary_for_shading = np.nan_to_num(center_u1_raw, nan=y_min, neginf=y_min, posinf=y_max)
    boundary_for_shading = np.clip(boundary_for_shading, y_min, y_max)
    x_fill = np.concatenate(([x_left], u_center, [x_right]))
    y_fill = np.concatenate(([boundary_for_shading[0]], boundary_for_shading, [boundary_for_shading[-1]]))
    ax_limit.fill_between(
        x_fill,
        y_min,
        y_fill,
        color="#f4a6a6",
        alpha=0.35,
        label="failure region (R < S)",
        zorder=0,
    )
    ax_limit.fill_between(
        x_fill,
        y_fill,
        y_max,
        color="#a8dca8",
        alpha=0.30,
        label="no-failure region (R >= S)",
        zorder=0,
    )

    ax_limit.plot(
        u_center,
        center_u1,
        color="#9467bd",
        linewidth=1.0,
        label="u1 on limit state (R = S)",
    )
    if np.isfinite(result.beta_star) and np.all(np.isfinite(result.alpha)):
        u_star = -np.asarray(result.alpha, dtype=float) * float(result.beta_star)
        ax_limit.scatter(
            [float(u_star[1])],
            [float(u_star[0])],
            color="#d62728",
            marker="x",
            s=64,
            label="design point",
        )
    ax_limit.set_title("Limit-State Signal in U-Space")
    ax_limit.set_xlabel("u2")
    ax_limit.set_ylabel("u1")
    ax_limit.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_limit.legend(loc="best")

    if result.u_bounds_used is not None:
        u_min, u_max = result.u_bounds_used
        ax_limit.axvline(float(u_min), color="#555555", linestyle=":", linewidth=0.8)
        ax_limit.axvline(float(u_max), color="#555555", linestyle=":", linewidth=0.8)

    fig.text(
        0.01,
        0.01,
        (f"converged={result.converged}, pf={result.pf:.6e}, beta_pf={result.beta_pf:.6f}"),
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    output: dict[str, np.ndarray | float] = {
        "u_left": u_left,
        "u_right": u_right,
        "u_center": u_center,
        "depth": depth,
        "prob_weight": prob_weight,
        "failure_cdf_center": failure_cdf_center,
        "local_pf_contribution": local_pf,
        "error_estimate": error_estimate,
        "u1_on_limit_state": center_u1,
        "center_u1": center_u1,
        "center_u1_shading": y_fill,
        "failure_region_ymin": float(y_min),
        "failure_region_ymax": float(y_max),
        "pf": float(result.pf),
    }
    return fig, axes_arr, output


def prepare_failure_histogram(
    result: IntegrationResult,
    bin_edges: np.ndarray | list[float],
    *,
    conditional: bool = False,
    solicitation_distribution: ot.Distribution | None = None,
) -> dict[str, np.ndarray]:
    """Aggregate failure weights over solicitation (water-level) bins.

    Parameters
    ----------
    result : IntegrationResult
        Reliability outcome whose failure samples will be binned.
    bin_edges : np.ndarray | list[float]
        Monotone sequence of bin edges in water-level coordinates.
    conditional : bool
        When ``True``, compute ``P(R < S | S in bin)`` by dividing each
        failure-weight bin by the corresponding solicitation probability mass.
        Defaults to ``False``.
    solicitation_distribution : ot.Distribution | None
        Distribution or hazard-derived distribution describing solicitation.
        Required when ``conditional`` is ``True``. Defaults to ``None``.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with at least ``bin_edges`` and ``failure_mass`` entries.
        ``conditional_failure`` is included when ``conditional`` is ``True``.

    Raises
    ------
    ValueError
        If failure samples lack solicitation levels, bins are invalid, or the
        bins do not cover all failure weight (sum differs from ``pf``).
    """
    weights = result.failure_samples.weights
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional sequence with at least two points.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")

    if weights.size == 0:
        failure_mass = np.zeros(edges.size - 1, dtype=float)
        data: dict[str, np.ndarray] = {
            "bin_edges": edges,
            "failure_mass": failure_mass,
        }
        if conditional:
            if solicitation_distribution is None:
                raise ValueError("solicitation_distribution is required for conditional probabilities.")
            cdf_vals = np.array(solicitation_distribution.computeCDF(edges[:, np.newaxis])).flatten()
            bin_probs = np.diff(cdf_vals)
            conditional_vals = np.divide(
                failure_mass,
                bin_probs,
                out=np.zeros_like(failure_mass),
                where=bin_probs > 0.0,
            )
            data["conditional_failure"] = conditional_vals
        return data

    water_levels = result.failure_solicitation_levels()
    if water_levels is None:
        if result.failure_samples.hazard_levels is not None:
            water_levels = result.failure_samples.hazard_levels
        else:
            raise ValueError("Failure samples do not expose solicitation levels.")

    failure_mass, hist_edges = np.histogram(water_levels, bins=edges, weights=weights)
    total_mass = float(failure_mass.sum())
    if not np.isclose(total_mass, result.pf, rtol=1e-12, atol=1e-15):
        raise ValueError(
            "Histogram bins do not cover all failure mass; "
            f"sum={total_mass:.6e}, pf={result.pf:.6e}. Adjust the bin range."
        )

    data: dict[str, np.ndarray] = {
        "bin_edges": hist_edges,
        "failure_mass": failure_mass,
    }

    if conditional:
        if solicitation_distribution is None:
            raise ValueError("solicitation_distribution is required for conditional probabilities.")
        cdf_vals = np.array(solicitation_distribution.computeCDF(hist_edges[:, np.newaxis])).flatten()
        bin_probs = np.diff(cdf_vals)
        conditional_vals = np.divide(
            failure_mass,
            bin_probs,
            out=np.zeros_like(failure_mass),
            where=bin_probs > 0.0,
        )
        data["conditional_failure"] = conditional_vals

    return data


def plot_failure_histogram(
    result: IntegrationResult,
    bin_edges: np.ndarray | list[float],
    *,
    conditional: bool = False,
    solicitation_distribution: ot.Distribution | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
) -> tuple[Figure, Axes, dict[str, np.ndarray]]:
    """Render a bar chart of failure probability versus water level.

    Parameters
    ----------
    result : IntegrationResult
        Reliability outcome supplying the failure samples.
    bin_edges : np.ndarray | list[float]
        Water-level bin edges passed through to :func:`prepare_failure_histogram`.
    conditional : bool
        When ``True``, plot conditional failure probabilities per bin; otherwise
        plot the absolute failure probability mass within each bin. Defaults to
        ``False``.
    solicitation_distribution : ot.Distribution | None
        Required when ``conditional`` is ``True`` so bin probability masses can
        be computed. Defaults to ``None``.
    ax : Axes | None
        Axes receiving the plot. A new figure/axes pair is created when omitted.
        Defaults to ``None``.
    figsize : tuple[float, float]
        Size for the fallback figure when ``ax`` is ``None``. Defaults to
        ``(7.0, 4.0)``.

    Returns
    -------
    Figure, Axes, dict[str, np.ndarray]
        Handles to the figure/axes plus the histogram data dict returned by
        :func:`prepare_failure_histogram`.
    """
    hist_data = prepare_failure_histogram(
        result,
        bin_edges,
        conditional=conditional,
        solicitation_distribution=solicitation_distribution,
    )
    edges = hist_data["bin_edges"]
    values = hist_data["conditional_failure"] if conditional else hist_data["failure_mass"]
    widths = np.diff(edges)
    centers = edges[:-1] + 0.5 * widths

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.bar(
        centers,
        values,
        width=widths,
        align="center",
        edgecolor="black",
        color="#1f77b4",
        alpha=0.85,
    )

    design_point = result.design_point_physical
    if design_point is not None and np.isfinite(design_point.solicitation):
        x_design = float(design_point.solicitation)
        ax.axvline(
            x_design,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label="design point (solicitation)",
        )
        ax.legend(loc="best")

    ax.set_xlabel("Water level")
    ax.set_ylabel("Pf per bin" if not conditional else "P(R < S | bin)")
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    if not conditional:
        ax.set_title("Failure probability by water-level bin")
    else:
        ax.set_title("Conditional failure probability by water-level bin")

    fig.tight_layout()
    return fig, ax, hist_data
