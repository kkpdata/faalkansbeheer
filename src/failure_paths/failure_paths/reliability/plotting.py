from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .core import ReliabilityIntegrator


class IntegrationGridPlotter:
    """Render the integration grid of a ReliabilityIntegrator instance."""

    def __init__(self, integrator: ReliabilityIntegrator) -> None:
        self.integrator = integrator

    def plot(
        self,
        ax: Axes | None = None,
        *,
        figsize: tuple[float, float] = (6.0, 6.0),
        limit_points: int = 1025,
    ) -> tuple[Figure, Axes]:
        """Visualize the coarse and refined U-grid along with the z=0 curve.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axes receiving the drawing. A fresh figure/axes pair is created when omitted.
        figsize : tuple[float, float]
            Size used for the fallback figure when ``ax`` is ``None``.
        limit_points : int
            Number of samples used to draw the ``z = 0`` curve in U-space.

        Returns
        -------
        Figure
            Matplotlib figure containing the visualization.
        Axes
            Axes instance on which the plot was rendered.
        """
        integrator = self.integrator
        grid = integrator._compute_distribution_grid()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        failure_color = "#c0392b"
        safe_color = "#1e8449"
        cmap = ListedColormap([failure_color, safe_color])
        cmap.set_bad(alpha=0.0)
        norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

        coarse_values = np.full(grid.fail_mask.shape, np.nan, dtype=float)
        coarse_values[grid.fail_mask] = 0.0
        coarse_values[grid.safe_mask] = 1.0

        ax.pcolormesh(
            grid.u_edges,
            grid.u_edges,
            coarse_values.T,
            cmap=cmap,
            norm=norm,
            shading="auto",
            zorder=1,
        )

        u_limits = (integrator.config.u_min, integrator.config.u_max)
        ax.vlines(grid.u_edges, *u_limits, color="#d0d0d0", linewidth=0.4, zorder=3)
        ax.hlines(grid.u_edges, *u_limits, color="#d0d0d0", linewidth=0.4, zorder=3)

        if (
            grid.subcell_fail_mask is not None
            and grid.subcell_weights is not None
            and grid.u1_sub_edges is not None
            and grid.u2_sub_edges is not None
        ):
            for idx in range(grid.mixed_indices.shape[0]):
                sub_values = np.where(grid.subcell_fail_mask[idx], 0.0, 1.0)
                ax.pcolormesh(
                    grid.u1_sub_edges[idx],
                    grid.u2_sub_edges[idx],
                    sub_values.T,
                    cmap=cmap,
                    norm=norm,
                    shading="auto",
                    zorder=2,
                )

            refine_segments: list[list[tuple[float, float]]] = []
            for idx, (i_cell, j_cell) in enumerate(grid.mixed_indices):
                x_edges = grid.u1_sub_edges[idx]
                y_edges = grid.u2_sub_edges[idx]
                x_inner = x_edges[1:-1]
                y_inner = y_edges[1:-1]

                y_bottom = grid.u_edges[j_cell]
                y_top = grid.u_edges[j_cell + 1]
                for x in x_inner:
                    refine_segments.append([(x, y_bottom), (x, y_top)])

                x_left = grid.u_edges[i_cell]
                x_right = grid.u_edges[i_cell + 1]
                for y in y_inner:
                    refine_segments.append([(x_left, y), (x_right, y)])

            if refine_segments:
                ax.add_collection(LineCollection(refine_segments, colors="#666666", linewidths=0.4, zorder=4))

        u_line = np.linspace(integrator.config.u_min, integrator.config.u_max, max(3, limit_points))
        u2_line = integrator._limit_state_curve(u_line)
        mask = np.isfinite(u2_line)
        if np.any(mask):
            ax.plot(u_line[mask], u2_line[mask], color="black", linewidth=1.4, label="z = 0", zorder=5)

        ax.set_xlabel("$u_R$")
        ax.set_ylabel("$u_S$")
        ax.set_xlim(u_limits)
        ax.set_ylim(u_limits)
        ax.set_aspect("equal", adjustable="box")

        handles = [
            Patch(facecolor=failure_color, edgecolor="black", label="Failure"),
            Patch(facecolor=safe_color, edgecolor="black", label="Safe"),
            Line2D([0], [0], color="black", linewidth=1.4, label="z = 0"),
        ]
        ax.legend(handles=handles, loc="upper right")

        return fig, ax
