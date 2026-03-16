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
    axes: np.ndarray | None = None,
    *,
    figsize: tuple[float, float] = (13.0, 5.0),
    cmap: str = "viridis",
) -> tuple[Figure, np.ndarray, dict[str, np.ndarray | float]]:
    """Render a 1D adaptive-integration diagnostics dashboard in U-space.

    Parameters
    ----------
    result : IntegrationResult
        Reliability integration result containing an optional diagnostics trace.
    axes : np.ndarray | None
        Optional array-like container with exactly two axes:
        ``[summary_axis, limit_state_axis]``. When omitted, a new 1x2 layout is created.
    figsize : tuple[float, float]
        Figure size used when ``axes`` is ``None``.
    cmap : str
        Matplotlib colormap for the refinement-map error coloring.

    Returns
    -------
    tuple[Figure, np.ndarray, dict[str, np.ndarray | float]]
        Figure handle, two-axis array, and diagnostics arrays used for plotting.

    Raises
    ------
    ValueError
        If diagnostics trace data is unavailable or ``axes`` does not contain
        exactly two axes.
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

    cumulative_pf = np.cumsum(local_pf)
    radius_proxy = np.where(
        np.isfinite(center_u1_raw), np.sqrt(np.maximum(0.0, center_u1_raw**2 + u_center**2)), np.nan
    )

    if axes is None:
        fig, axes_arr = plt.subplots(1, 2, figsize=figsize)
        axes_arr = np.asarray(axes_arr, dtype=object).reshape(2)
    else:
        axes_arr = np.asarray(axes, dtype=object).reshape(-1)
        if axes_arr.size != 2:
            raise ValueError("axes must contain exactly two Matplotlib axes.")
        fig = axes_arr[0].figure

    ax_summary, ax_limit = axes_arr

    interval_width = np.maximum(u_right - u_left, 1e-12)
    bars = ax_summary.bar(
        u_center,
        local_pf,
        width=interval_width,
        align="center",
        color="#4c78a8",
        alpha=0.8,
        edgecolor="none",
        label="local Pf contribution",
    )
    ax_summary.plot(u_center, local_pf, color="#2f4b7c", linewidth=0.9)
    ax_summary.set_title("Integration Summary (Contribution, Cumulative, Refinement)")
    ax_summary.set_xlabel("u2 interval center")
    ax_summary.set_ylabel("local Pf contribution")
    ax_summary.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    ax_cumulative = ax_summary.twinx()
    cumulative_line = ax_cumulative.plot(
        u_center,
        cumulative_pf,
        color="#2ca02c",
        linewidth=1.2,
        label="cumulative Pf",
    )[0]
    ax_cumulative.axhline(float(result.pf), color="#1f1f1f", linestyle=":", linewidth=1.0)
    ax_cumulative.set_ylabel("cumulative Pf")

    max_prob_weight = float(np.max(prob_weight)) if prob_weight.size > 0 else 0.0
    size_scale = 20.0 + 100.0 * (prob_weight / max(max_prob_weight, 1e-300))
    max_depth = float(np.max(depth)) if depth.size > 0 else 0.0
    depth_alpha = 0.35 + 0.65 * (depth / max(max_depth, 1.0))
    scatter = ax_summary.scatter(
        u_center,
        np.full_like(u_center, 0.97),
        c=error_estimate,
        s=size_scale,
        cmap=cmap,
        alpha=np.clip(depth_alpha, 0.35, 1.0),
        linewidths=0.3,
        edgecolors="#222222",
        transform=ax_summary.get_xaxis_transform(),
        zorder=4,
    )
    cbar = fig.colorbar(scatter, ax=ax_summary)
    cbar.set_label("interval error estimate")
    ax_summary.text(
        0.01,
        0.95,
        "marker size = prob_weight, marker alpha = depth",
        transform=ax_summary.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )

    ax_limit.plot(u_center, center_u1, color="#9467bd", linewidth=1.0, label="u1_eq(u2)")
    ax_limit.plot(u_center, radius_proxy, color="#ff7f0e", linewidth=1.0, linestyle="--", label="sqrt(u1^2+u2^2)")
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
    ax_limit.set_xlabel("u2 interval center")
    ax_limit.set_ylabel("u1_eq / radius proxy")
    ax_limit.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_limit.legend(loc="best")

    if result.u_bounds_used is not None:
        u_min, u_max = result.u_bounds_used
        for axis in (ax_summary, ax_limit):
            axis.axvline(float(u_min), color="#555555", linestyle=":", linewidth=0.8)
            axis.axvline(float(u_max), color="#555555", linestyle=":", linewidth=0.8)

    ax_summary.legend(handles=[bars, cumulative_line], loc="upper right")

    convergence_reason = result.convergence_reason if result.convergence_reason is not None else "unknown"
    trunc_err = result.truncation_pf_error_bound if result.truncation_pf_error_bound is not None else float("nan")
    adaptive_err = (
        result.adaptive_remaining_pf_error if result.adaptive_remaining_pf_error is not None else float("nan")
    )
    fig.text(
        0.01,
        0.01,
        (
            f"convergence={convergence_reason}, "
            f"pf={result.pf:.6e}, "
            f"beta_pf={result.beta_pf:.6f}, "
            f"truncation_pf_error_bound={trunc_err:.3e}, "
            f"adaptive_pf_error={adaptive_err:.3e}, "
            f"u_bounds={result.u_bounds_used}"
        ),
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
        "cumulative_pf": cumulative_pf,
        "error_estimate": error_estimate,
        "center_u1": center_u1,
        "radius_proxy": radius_proxy,
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

    water_levels = result.failure_water_levels()
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
