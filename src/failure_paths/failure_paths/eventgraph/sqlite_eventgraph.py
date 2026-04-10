from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import networkx as nx

from .eventgraph import EventGraph
from .models import EventTable, FrequencyTable, MetadataTable
from .sqlite_store import EventGraphStore, ScenarioInfo


class SqliteEventGraph(EventGraph):
    """Event graph that loads its state from a previously ingested SQLite database."""

    metadata: MetadataTable
    graph: nx.DiGraph
    graph_events: EventTable
    freq_tables: dict[str, FrequencyTable]
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
        scenario_id : int | None
            Explicit scenario id to load. When omitted, the ``section``,
            ``source_path``, and ``sheet_name`` triple must be supplied. Defaults
            to ``None``.
        section : str | None, optional
            Section identifier used when ``scenario_id`` is not provided.
        source_path : str | None, optional
            Source workbook path used when ``scenario_id`` is not provided.
        sheet_name : str | None, optional
            Sheet identifier used when ``scenario_id`` is not provided.

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
            freq_tables = store.read_frequency_tables(resolved_id)
            return cls(
                metadata=metadata,
                graph=graph,
                graph_events=events,
                freq_tables=freq_tables,
                scenario_id=resolved_id,
            )
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
        section : str | None, optional
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

    @staticmethod
    def available_sections(db_path: str | Path) -> list[str]:
        """List distinct sections stored in the SQLite database.

        Parameters
        ----------
        db_path : str | Path
            SQLite database path.

        Returns
        -------
        list[str]
            Distinct section names sorted in ascending order.
        """
        store = EventGraphStore(db_path, read_only=True)
        try:
            return store.list_sections()
        finally:
            store.close()

    @staticmethod
    def available_node_names(
        db_path: str | Path,
        *,
        sections: str | Sequence[str] | None = None,
        node_types: str | Sequence[str] | None = None,
    ) -> list[str]:
        """List distinct node descriptions, optionally filtered by section(s) and node type.

        Parameters
        ----------
        db_path : str | Path
            SQLite database path.
        sections : str | Sequence[str] | None, optional
            Optional section filter. Accepts a single section string, a
            sequence of section strings, or ``None`` for all sections.
        node_types : str | Sequence[str] | None, optional
            Optional node-type filter. Accepts a single node-type string, a
            sequence of node-type strings, or ``None`` for all node types.

        Returns
        -------
        list[str]
            Distinct node descriptions sorted in ascending order.
        """
        store = EventGraphStore(db_path, read_only=True)
        try:
            return store.list_node_names(
                sections=sections,
                node_types=node_types,
            )
        finally:
            store.close()
