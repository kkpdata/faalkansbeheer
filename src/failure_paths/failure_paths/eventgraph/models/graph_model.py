from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Typed node payload used in the event graph.

    Attributes
    ----------
    node_id : tuple[int, int]
        Stable node identifier encoded as ``(faalpad_id, knoop_id)``.
    description : str | None
        Human-readable node label. Can be ``None`` when no description is
        available in the source data.
    node_type : str
        Node category used by graph logic and rendering, for example
        ``"start_node"``, ``"scenario_node"``, ``"event_node"``, or
        ``"failure_node"``.
    type_name : str | None
        Optional subtype/category label for scenario nodes.
    """

    node_id: tuple[int, int]
    description: str | None
    node_type: str
    type_name: str | None = None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Typed representation of a directed edge between two graph nodes.

    Attributes
    ----------
    source : tuple[int, int]
        Identifier of the source node.
    target : tuple[int, int]
        Identifier of the destination node.
    """

    source: tuple[int, int]
    target: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FailurePath(Sequence[tuple[int, int]]):
    """Immutable ordered sequence of node identifiers describing one path.

    The path is represented as a tuple of node ids and exposed through the
    :class:`collections.abc.Sequence` interface.

    Attributes
    ----------
    nodes : tuple[tuple[int, int], ...]
        Ordered node identifiers from path source to terminal failure node.
    """

    nodes: tuple[tuple[int, int], ...]

    def __getitem__(self, index: int | slice) -> tuple[int, int] | tuple[tuple[int, int], ...]:
        """Return one node id or a slice of node ids from the path.

        Parameters
        ----------
        index : int | slice
            Position or slice to retrieve from :attr:`nodes`.

        Returns
        -------
        tuple[int, int] | tuple[tuple[int, int], ...]
            Single node identifier for integer lookup, or a tuple of node
            identifiers for slice lookup.
        """
        return self.nodes[index]

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of node identifiers in the path.

        Returns
        -------
        int
            Path length measured as number of nodes.
        """
        return len(self.nodes)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """Iterate over node identifiers in path order.

        Returns
        -------
        Iterator[tuple[int, int]]
            Iterator yielding ``(faalpad_id, knoop_id)`` tuples.
        """
        return iter(self.nodes)


@dataclass(slots=True)
class FailurePathProbabilities:
    """Attach path-level probability matrices to a :class:`FailurePath`.

    Attributes
    ----------
    path : FailurePath
        Ordered node identifiers describing the path.
    water_levels : np.ndarray
        One-dimensional water-level axis used for all probability arrays.
    node_probabilities : pd.DataFrame
        Per-node probabilities for each water level. Rows correspond to
        ``water_levels`` and columns correspond to ``path.nodes``.
    cumulative_probabilities : pd.DataFrame
        Cumulative probability matrix over regular (non-scenario) nodes along
        the path.
    combined_path_probabilities : pd.Series | None
        Per-path combined curve including scenario/start-node effects.
        When available, this is the canonical per-path contribution used in
        the overall combined scenario curve. Index is aligned to
        ``water_levels``.
    """

    path: FailurePath
    water_levels: np.ndarray
    node_probabilities: pd.DataFrame
    cumulative_probabilities: pd.DataFrame
    combined_path_probabilities: pd.Series | None = None
