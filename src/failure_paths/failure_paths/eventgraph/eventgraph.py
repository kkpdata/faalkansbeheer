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
from pydantic import BaseModel, ConfigDict

from .models import (
    EventTable,
    FailurePath,
    FailurePathProbabilities,
    MetadataTable,
)


class EventGraph(BaseModel, ABC):
    """Coordinate tabular data and graph rendering logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: MetadataTable
    graph: networkx.DiGraph
    graph_events: EventTable

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
        wrap_width : int, default=28
            Maximum number of characters per line for node labels.
        output_path : str | Path, default="output/event_tree.png"
            File path (without extension) used by Graphviz when writing the plot.
        view : bool, default=False
            When ``True`` the rendered file is opened using Graphviz' viewer.
        water_level : float | None, default=None
            Optional water level used to annotate Pf/Beta values on the nodes.
        graph_attr, node_attr, edge_attr, event_attr : dict[str, str] | None
            Optional Graphviz attribute overrides for the global graph, nodes,
            edges, or event-style colours.

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
        graph_attr, node_attr, edge_attr, event_attr : dict[str, str] | None
            Attribute overrides for Graphviz entities.

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
        start_nodes : list[tuple[int, int]] | None, default=None
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

    def get_failure_path_probabilities(
        self,
        water_levels: float | Sequence[float],
        *,
        start_nodes: list[tuple[int, int]] | None = None,
        start_node_pf: float = 1.0,
    ) -> list[FailurePathProbabilities]:
        """Annotate each simple path with Pf and cumulative probabilities.

        Parameters
        ----------
        water_levels : float | Sequence[float]
            Single water level or array of levels where Pf/Beta should be evaluated.
        start_nodes : list[tuple[int, int]] | None, default=None
            Custom start nodes passed to :meth:`get_failure_paths`.
        start_node_pf : float, default=1.0
            Probability assigned to start nodes before multiplying downstream Pf.

        Returns
        -------
        list[FailurePathProbabilities]
            Failure paths decorated with per-node and cumulative probabilities.
        """
        levels = np.atleast_1d(np.array(water_levels, dtype=float))
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("water_levels must be a non-empty scalar or 1D sequence")

        failure_paths = self.get_failure_paths(start_nodes=start_nodes)
        results: list[FailurePathProbabilities] = []
        fc_data = []
        for failure_path in failure_paths:
            nodes = failure_path.nodes
            prob_matrix = np.full((len(levels), len(nodes)), np.nan, dtype=float)

            for idx, nid in enumerate(nodes):
                node = self.graph.nodes[nid]
                if node["node_type"] == "start_node":
                    prob_matrix[:, idx] = start_node_pf
                else:
                    prob_matrix[:, idx] = self.graph_events.get_event_probs(nid, levels, as_beta=False)

            with np.errstate(divide="ignore"):
                cum_prob_matrix = np.log(prob_matrix)
            cum_prob_matrix = np.cumsum(cum_prob_matrix, axis=1)
            cum_prob_matrix = np.exp(cum_prob_matrix)

            # Save last column
            fc_data.append(cum_prob_matrix[:, [-1]])

            results.append(
                FailurePathProbabilities(
                    path=failure_path,
                    water_levels=levels.copy(),
                    node_probabilities=pd.DataFrame(prob_matrix, index=levels, columns=list(nodes)),
                    cumulative_probabilities=pd.DataFrame(cum_prob_matrix, index=levels, columns=list(nodes)),
                )
            )

        # use math.fsum for accurate row-wise summation
        fc_data = np.hstack(fc_data)
        fc_comb = pd.Series(index=levels, data=np.array([math.fsum(row) for row in fc_data], dtype=float))

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
