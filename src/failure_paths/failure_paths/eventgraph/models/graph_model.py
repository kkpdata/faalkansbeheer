from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Typed representation of a node used in the event graph."""

    node_id: tuple[int, int]
    description: str | None
    node_type: str
    type_name: str | None = None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """Typed representation of a directed edge between two nodes."""

    source: tuple[int, int]
    target: tuple[int, int]


@dataclass(frozen=True, slots=True)
class FailurePath(Sequence[tuple[int, int]]):
    """Immutable sequence of node identifiers describing a failure path."""

    nodes: tuple[tuple[int, int], ...]

    def __getitem__(self, index):
        return self.nodes[index]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.nodes)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.nodes)


@dataclass(slots=True)
class FailurePathProbabilities:
    """Attach Pf matrices to a :class:`FailurePath`."""

    path: FailurePath
    water_levels: np.ndarray
    node_probabilities: pd.DataFrame
    cumulative_probabilities: pd.DataFrame
