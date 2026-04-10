from __future__ import annotations

import math
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import graphviz
import networkx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ..common.prob import beta_from_pf, cumulative_beta_equivalent_ot, pf_from_beta
from .models import (
    EventTable,
    FailurePath,
    FailurePathProbabilities,
    FrequencyTable,
    MetadataTable,
)


class EventGraph(BaseModel, ABC):
    """Coordinate tabular data and graph rendering logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: MetadataTable
    graph: networkx.DiGraph
    graph_events: EventTable
    freq_tables: dict[str, FrequencyTable] = Field(default_factory=dict)

    @classmethod
    @abstractmethod
    def load(cls) -> Self:
        """Load the graph data and construct the directed graph."""
        pass

    def plot(
        self,
        wrap_width: int = 28,
        output_path: str | Path = "output/event_tree.png",
        view: bool = False,
        water_level: float | None = None,
        graph_attr: dict[str, str] | None = None,
        node_attr: dict[str, str] | None = None,
        edge_attr: dict[str, str] | None = None,
        event_attr: dict[str, str] | None = None,
    ) -> tuple[Path, graphviz.Digraph]:
        """Render the event graph using Graphviz with wrapped labels.

        Parameters
        ----------
        wrap_width : int
            Maximum number of characters per line for node labels. Defaults to ``28``.
        output_path : str | Path
            File path (without extension) used by Graphviz when writing the plot.
            Defaults to ``"output/event_tree.png"``.
        view : bool
            When ``True`` the rendered file is opened using Graphviz' viewer.
            Defaults to ``False``.
        water_level : float | None
            Optional water level used to annotate Pf/Beta values on the nodes.
            Defaults to ``None``.
        graph_attr : dict[str, str] | None, optional
            Graphviz attribute overrides for the global graph.
        node_attr : dict[str, str] | None, optional
            Graphviz attribute overrides for nodes.
        edge_attr : dict[str, str] | None, optional
            Graphviz attribute overrides for edges.
        event_attr : dict[str, str] | None, optional
            Graphviz attribute overrides for event-style colours.

        Returns
        -------
        tuple[Path, graphviz.Digraph]
            Path to the rendered file and the Graphviz digraph instance.
        """
        dot = self._build_graphviz(
            wrap_width=wrap_width,
            water_level=water_level,
            graph_attr=graph_attr,
            node_attr=node_attr,
            edge_attr=edge_attr,
            event_attr=event_attr,
        )
        output_path = Path(output_path)
        result_path = dot.render(
            filename=str(output_path.parent / output_path.stem),
            format=output_path.suffix[1:],
            cleanup=True,
            view=view,
        )
        return Path(result_path), dot

    def _build_graphviz(
        self,
        wrap_width: int,
        water_level: float | None,
        graph_attr: dict[str, str] | None,
        node_attr: dict[str, str] | None,
        edge_attr: dict[str, str] | None,
        event_attr: dict[str, str] | None,
    ) -> graphviz.Digraph:
        """Create a Graphviz diagram matching the in-memory event graph.

        Parameters
        ----------
        wrap_width : int
            Maximum label line length.
        water_level : float | None
            Optional water level used to annotate Pf/Beta.
        graph_attr : dict[str, str] | None, optional
            Attribute overrides for Graphviz graph-level settings.
        node_attr : dict[str, str] | None, optional
            Attribute overrides for Graphviz nodes.
        edge_attr : dict[str, str] | None, optional
            Attribute overrides for Graphviz edges.
        event_attr : dict[str, str] | None, optional
            Attribute overrides for event-style colours.

        Returns
        -------
        graphviz.Digraph
            Populated diagram ready for rendering.
        """
        base_graph_attr = {"rankdir": "LR", "nodesep": "0.6", "ranksep": "1.0"}
        base_node_attr = {
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#d7e8ff",
            "color": "#4a708b",
            "fontname": "Helvetica",
            "fontsize": "10",
        }
        base_edge_attr = {"color": "#4a708b", "arrowhead": "normal"}
        base_event_attr = {
            "failure_fill": "#f8d7da",
            "failure_border": "#b0413e",
            "scen_fill": "#b2ffb6",
            "scen_border": "#3c8500",
            "start_fill": "#fffab2",
            "start_border": "#857A00",
        }

        graph_attrs = {**base_graph_attr, **(graph_attr or {})}
        node_attrs = {**base_node_attr, **(node_attr or {})}
        edge_attrs = {**base_edge_attr, **(edge_attr or {})}
        event_attrs = {**base_event_attr, **(event_attr or {})}

        dot = graphviz.Digraph(
            "EventTree",
            graph_attr=graph_attrs,
            node_attr=node_attrs,
            edge_attr=edge_attrs,
        )

        for nid, node in self.graph.nodes.items():
            pf = None
            beta = None
            if water_level is not None and node["node_type"] != "start_node":
                pf = self.graph_events.get_event_prob(nid, water_level, as_beta=False)
                beta = self.graph_events.get_event_prob(nid, water_level, as_beta=True)
            label_text = node.get("description") if node.get("description") else str(node.get("nodeid"))

            wrapped = EventGraph._format_label(label_text, wrap_width)
            if pf is not None:
                if pf in (0.0, 1.0):
                    wrapped += f"\\n\\nh={water_level:.2f}\\npf={pf:.1f}"
                elif 0.999 <= pf < 1.0:
                    wrapped += f"\\n\\nh={water_level:.2f}\\npf=1 - {(1 - pf):.2e}"
                else:
                    wrapped += f"\\n\\nh={water_level:.2f}\\npf={pf:.2e}"
            if beta is not None:
                wrapped += f"\\nβ={beta:.2f}"

            node_kwargs: dict[str, str] = {}
            if node["node_type"] == "failure_node":
                node_kwargs["fillcolor"] = event_attrs["failure_fill"]
                node_kwargs["color"] = event_attrs["failure_border"]
            elif node["node_type"] == "scenario_node":
                node_kwargs["fillcolor"] = event_attrs["scen_fill"]
                node_kwargs["color"] = event_attrs["scen_border"]
            elif node["node_type"] == "start_node":
                node_kwargs["fillcolor"] = event_attrs["start_fill"]
                node_kwargs["color"] = event_attrs["start_border"]
            dot.node(str(nid), label=wrapped, **node_kwargs)

        for edge in self.graph.edges:
            dot.edge(str(edge[0]), str(edge[1]))

        return dot

    def get_failure_paths(
        self,
        start_nodes: list[tuple[int, int]] | None = None,
    ) -> list[FailurePath]:
        """Return all unique simple paths that terminate in failure nodes.

        Parameters
        ----------
        start_nodes : list[tuple[int, int]] | None, optional
            Optional list of graph node identifiers to use as path sources.
            When omitted, nodes with zero in-degree are treated as sources.

        Returns
        -------
        list[FailurePath]
            Sorted list of unique simple paths reaching failure nodes.
        """
        if start_nodes is None:
            start_nodes = [node for node, indeg in self.graph.in_degree() if indeg == 0]

        failure_nodes = [
            node_id for node_id, data in self.graph.nodes(data=True) if data.get("node_type") == "failure_node"
        ]

        paths: list[FailurePath] = []
        seen = set()
        for source in start_nodes:
            for target in failure_nodes:
                if not networkx.has_path(self.graph, source, target):
                    continue
                for path in networkx.all_simple_paths(self.graph, source=source, target=target):
                    key = tuple(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    paths.append(FailurePath(nodes=key))

        return paths

    @staticmethod
    def aggregate_combined_path_probabilities(
        path_results: list[FailurePathProbabilities],
        *,
        include_nodes: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
        ensure_monotone: bool = False,
        clip: bool = False,
    ) -> np.ndarray:
        """Aggregate per-path combined curves with optional node filtering.

        Parameters
        ----------
        path_results : list[FailurePathProbabilities]
            Path-level probability outputs returned by
            :meth:`get_failure_path_probabilities`.
        include_nodes : tuple[int, int] | Sequence[tuple[int, int]] | None, optional
            Optional node filter. Accepts either one node id or multiple node
            ids. When provided, only paths containing at least one of those
            node ids are included in the sum.
        ensure_monotone : bool, optional
            Enforce non-decreasing values over water levels.
        clip : bool, optional
            Clip aggregated probabilities to the ``[0, 1]`` interval.

        Returns
        -------
        np.ndarray
            One-dimensional aggregated probability curve aligned to
            ``path_results[0].water_levels``.

        Raises
        ------
        ValueError
            If no paths are provided, if no path matches ``include_nodes``,
            or if combined-path payloads are missing.
        """
        if len(path_results) == 0:
            raise ValueError("No failure-path probabilities are available.")

        include_node_values: set[tuple[int, int]] | None
        if include_nodes is None:
            include_node_values = None
        elif (
            isinstance(include_nodes, tuple)
            and len(include_nodes) == 2
            and all(isinstance(v, (int, np.integer)) for v in include_nodes)
        ):
            include_node_values = {(int(include_nodes[0]), int(include_nodes[1]))}
        else:
            include_node_values = {(int(node[0]), int(node[1])) for node in include_nodes}

        selected_results = [
            result
            for result in path_results
            if include_node_values is None or include_node_values.intersection(result.path.nodes)
        ]
        if len(selected_results) == 0:
            if include_node_values is None:
                raise ValueError("No failure paths available for combined-path aggregation.")
            raise ValueError(f"Node filter {include_nodes!r} is not part of any failure path.")

        if any(result.combined_path_probabilities is None for result in selected_results):
            raise ValueError("Missing combined path probabilities for a failure path result.")

        curve_matrix = np.column_stack(
            [
                np.asarray(result.combined_path_probabilities.to_numpy(dtype=float), dtype=float)
                for result in selected_results
            ]
        )
        aggregated = np.array([math.fsum(row) for row in curve_matrix], dtype=float)

        if ensure_monotone:
            for i in range(1, len(aggregated)):
                if aggregated[i] < aggregated[i - 1]:
                    aggregated[i] = aggregated[i - 1]
        if clip:
            aggregated = np.clip(aggregated, 0.0, 1.0)
        return aggregated

    @staticmethod
    def aggregate_weighted_scenario_curves(
        curves: Sequence[np.ndarray | Sequence[float]],
        weights: Sequence[float],
        *,
        require_sum_one: bool = True,
        atol: float = 1e-9,
    ) -> np.ndarray:
        """Compute a weighted sum across scenario curves.

        Parameters
        ----------
        curves : Sequence[np.ndarray | Sequence[float]]
            One-dimensional scenario curves, all sharing the same length.
        weights : Sequence[float]
            Scenario weights aligned with ``curves``.
        require_sum_one : bool, optional
            Require the sum of ``weights`` to be approximately ``1``.
        atol : float, optional
            Absolute tolerance used when checking ``sum(weights) == 1``.

        Returns
        -------
        np.ndarray
            Weighted curve with the same length as each input curve.

        Raises
        ------
        ValueError
            If inputs are empty, lengths mismatch, weights are non-finite,
            weight sums are invalid, or curve lengths are inconsistent.
        """
        if len(curves) == 0:
            raise ValueError("curves must contain at least one scenario curve.")
        if len(curves) != len(weights):
            raise ValueError("curves and weights must have the same length.")

        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must be finite numeric values.")
        if require_sum_one and not np.isclose(float(weight_array.sum()), 1.0, rtol=0.0, atol=atol):
            raise ValueError(f"Scenario weights must sum to 1. Got {float(weight_array.sum()):.12g}.")

        curve_arrays = [np.asarray(curve, dtype=float).reshape(-1) for curve in curves]
        curve_lengths = {curve.shape[0] for curve in curve_arrays}
        if len(curve_lengths) != 1:
            raise ValueError("All scenario curves must have the same length.")

        curve_matrix = np.column_stack(curve_arrays)
        weighted_matrix = curve_matrix * weight_array[np.newaxis, :]
        return np.array([math.fsum(row) for row in weighted_matrix], dtype=float)

    def get_failure_path_probabilities(
        self,
        water_levels: float | Sequence[float],
        *,
        start_nodes: list[tuple[int, int]] | None = None,
        start_node_pf: float = 1.0,
        ensure_monotone: bool = True,
    ) -> tuple[pd.Series, list[FailurePathProbabilities]]:
        """Annotate each simple path with Pf and cumulative probabilities.

        Parameters
        ----------
        water_levels : float | Sequence[float]
            Single water level or array of levels where Pf/Beta should be evaluated.
        start_nodes : list[tuple[int, int]] | None, optional
            Custom start nodes passed to :meth:`get_failure_paths`.
        start_node_pf : float, optional
            Probability assigned to start nodes before multiplying downstream Pf.
        ensure_monotone : bool
            Ensure resulting fragility curve is monotone

        Returns
        -------
        tuple[pd.Series, list[FailurePathProbabilities]]
            ``(fc_comb, path_results)`` where ``fc_comb`` is the aggregated
            scenario fragility curve and ``path_results`` are the path-level
            probability payloads.

        Raises
        ------
        ValueError
            If ``water_levels`` is empty or not a 1D sequence.
        """
        levels = np.atleast_1d(np.array(water_levels, dtype=float))
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("water_levels must be a non-empty scalar or 1D sequence")

        start_beta = float(beta_from_pf(start_node_pf, tail="upper"))

        failure_paths = self.get_failure_paths(start_nodes=start_nodes)
        results: list[FailurePathProbabilities] = []
        for failure_path in failure_paths:
            nodes = failure_path.nodes
            beta_matrix = np.full((len(levels), len(nodes)), np.nan, dtype=float)
            prob_matrix = beta_matrix.copy()

            # Fill probabilities for each node
            scen_idxs = []
            cum_idxs = []
            for idx, nid in enumerate(nodes):
                node = self.graph.nodes[nid]
                if node["node_type"] in ["start_node", "scenario_node"]:
                    scen_idxs.append(idx)
                else:
                    cum_idxs.append(idx)

                if node["node_type"] == "start_node":
                    beta_matrix[:, idx] = start_beta
                else:
                    beta_matrix[:, idx] = self.graph_events.get_event_probs(nid, levels, as_beta=True)
                prob_matrix[:, idx] = pf_from_beta(beta_matrix[:, idx], tail="upper")

            # Product of probabilities is sum of betas
            cum_prob_matrix = np.full_like(beta_matrix, np.nan)
            cum_beta_matrix = np.full_like(beta_matrix, np.nan)
            cum_beta_matrix[:, cum_idxs], _, cum_prob_matrix[:, cum_idxs] = cumulative_beta_equivalent_ot(
                beta_matrix[:, cum_idxs], axis=1
            )

            # Combine scenario columns and last column
            tot_cum = np.hstack([beta_matrix[:, scen_idxs], cum_beta_matrix[:, [-1]]])
            _, _, tot_cum = cumulative_beta_equivalent_ot(tot_cum, axis=1)

            results.append(
                FailurePathProbabilities(
                    path=failure_path,
                    water_levels=levels.copy(),
                    node_probabilities=pd.DataFrame(prob_matrix, index=levels, columns=list(nodes)),
                    cumulative_probabilities=pd.DataFrame(cum_prob_matrix, index=levels, columns=list(nodes)),
                    combined_path_probabilities=pd.Series(tot_cum[:, -1], index=levels, name=nodes[-1]),
                )
            )

        fc_comb = self.aggregate_combined_path_probabilities(results, ensure_monotone=ensure_monotone, clip=True)
        fc_comb = pd.Series(index=levels, data=fc_comb)

        return fc_comb, results

    @staticmethod
    def _format_label(text: str, width: int) -> str:
        """Wrap labels for Graphviz nodes.

        Parameters
        ----------
        text : str
            Label text to wrap.
        width : int
            Maximum characters per line.

        Returns
        -------
        str
            Wrapped label with newline separators.
        """
        lines = textwrap.wrap(str(text), width=width) or [text]
        return "\n".join(lines)
