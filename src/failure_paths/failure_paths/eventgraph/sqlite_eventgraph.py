from __future__ import annotations

from pathlib import Path
from typing import Self

import networkx as nx

from .eventgraph import EventGraph
from .models import EventTable, MetadataTable
from .sqlite_store import EventGraphStore, ScenarioInfo


class SqliteEventGraph(EventGraph):
    """Event graph that loads its state from a previously ingested SQLite database."""

    metadata: MetadataTable
    graph: nx.DiGraph
    graph_events: EventTable
    scenario_id: int | None = None

    @classmethod
    def load(
        cls,
        db_path: str | Path,
        *,
        scenario_id: int | None = None,
        section: str | None = None,
        source_path: str | None = None,
        sheet_name: str | None = None,
    ) -> Self:
        """Load a scenario from the SQLite store.

        Parameters
        ----------
        db_path : str | Path
            SQLite database created by :class:`EventGraphStore`.
        scenario_id : int | None, default=None
            Explicit scenario id to load. When omitted, the ``section``,
            ``source_path``, and ``sheet_name`` triple must be supplied.
        section, source_path, sheet_name : str | None
            Scenario identifiers used when ``scenario_id`` is not provided.

        Returns
        -------
        SqliteEventGraph
            Graph populated from the stored scenario rows.
        """
        store = EventGraphStore(db_path, read_only=True)
        try:
            scenario = store.fetch_scenario(
                scenario_id=scenario_id,
                section=section,
                source_path=source_path,
                sheet_name=sheet_name,
            )
            resolved_id = scenario.id
            metadata = MetadataTable.from_records(store.read_dataframe("metadata", resolved_id))
            events = EventTable.from_records(store.read_dataframe("events", resolved_id))
            graph = store.read_graph(resolved_id)
            return cls(metadata=metadata, graph=graph, graph_events=events, scenario_id=resolved_id)
        finally:
            store.close()

    @staticmethod
    def available_scenarios(
        db_path: str | Path,
        *,
        section: str | None = None,
    ) -> list[ScenarioInfo]:
        """List scenarios stored in the SQLite database.

        Parameters
        ----------
        db_path : str | Path
            SQLite database path.
        section : str | None, default=None
            Optional filter limiting the results to a single section.

        Returns
        -------
        list[ScenarioInfo]
            Metadata records describing each available scenario.
        """
        store = EventGraphStore(db_path, read_only=True)
        try:
            return store.list_scenarios(section=section)
        finally:
            store.close()
