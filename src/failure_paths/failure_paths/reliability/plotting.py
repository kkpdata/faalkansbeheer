from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import ReliabilityIntegrator
from .results import IntegrationResult


def plot_integration_grid(
    integrator: ReliabilityIntegrator,
    ax: Axes | None = None,
    *,
    figsize: tuple[float, float] = (6.0, 6.0),
    limit_points: int = 1025,
) -> tuple[Figure, Axes]:
    """Deprecated grid plotting API.

    Parameters
    ----------
    integrator : ReliabilityIntegrator
        Integrator whose grid and diagnostics will be visualized.
    ax : Axes | None
        Matplotlib axes receiving the drawing. A fresh figure/axes pair is created when omitted.
    figsize : tuple[float, float]
        Size used for the fallback figure when ``ax`` is ``None``.
    limit_points : int
        Number of samples used to draw the ``z = 0`` curve in U-space.

    Returns
    -------
    tuple[Figure, Axes]
        This deprecated function always raises and therefore does not return.

    Raises
    ------
    RuntimeError
        Always raised because the 2D grid integrator was removed in favor of the
        1D independent integration path.
    """
    raise RuntimeError(
        "plot_integration_grid is deprecated: the 2D grid integrator has been removed "
        "in favor of the 1D independent integration path."
    )


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
