from __future__ import annotations

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

"""
SQLite persistence helpers for Excel-derived failure-path scenarios.

This module centralises the schema definition, connection lifecycle, and the
serialization routines required to store tabular scenario data alongside the
networkx digraph representation.  Higher-level loaders (Excel/SQLite
EventGraphs) can rely on :class:`EventGraphStore` to keep inserts transactional
and efficient.
"""


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ScenarioInfo:
    id: int
    section: str
    source_path: str
    sheet_name: str
    start_node_label: str
    ingested_at: str
    tags: str | None


CREATE_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS scenarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        section TEXT NOT NULL,
        source_path TEXT NOT NULL,
        sheet_name TEXT NOT NULL,
        start_node_label TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        tags TEXT,
        UNIQUE(section, source_path, sheet_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metadata (
        scenario_id INTEGER PRIMARY KEY,
        dijkvaknummer INTEGER,
        Vaknaam TEXT,
        LENGTE_VAK REAL,
        TYPE_WATERKERING TEXT,
        Ondergrondscenario TEXT,
        ScenarioKans REAL,
        HR_locatie TEXT,
        FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        scenario_id INTEGER NOT NULL,
        Faalpad_ID INTEGER NOT NULL,
        Knoop_ID INTEGER NOT NULL,
        h REAL NOT NULL,
        Pf_h REAL,
        Beta_h REAL,
        PRIMARY KEY (scenario_id, Faalpad_ID, Knoop_ID, h),
        FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_nodes (
        scenario_id INTEGER NOT NULL,
        faalpad_id INTEGER NOT NULL,
        knoop_id INTEGER NOT NULL,
        description TEXT,
        node_type TEXT NOT NULL,
        type_name TEXT,
        PRIMARY KEY (scenario_id, faalpad_id, knoop_id),
        FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS graph_edges (
        scenario_id INTEGER NOT NULL,
        source_faalpad INTEGER NOT NULL,
        source_knoop INTEGER NOT NULL,
        target_faalpad INTEGER NOT NULL,
        target_knoop INTEGER NOT NULL,
        PRIMARY KEY (
            scenario_id,
            source_faalpad,
            source_knoop,
            target_faalpad,
            target_knoop
        ),
        FOREIGN KEY (
            scenario_id,
            source_faalpad,
            source_knoop
        ) REFERENCES graph_nodes(scenario_id, faalpad_id, knoop_id) ON DELETE CASCADE,
        FOREIGN KEY (
            scenario_id,
            target_faalpad,
            target_knoop
        ) REFERENCES graph_nodes(scenario_id, faalpad_id, knoop_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_scenario ON events (scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_scenario ON graph_nodes (scenario_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_scenario ON graph_edges (scenario_id)",
)


class EventGraphStore:
    """Lightweight helper that persists scenario data and graph artifacts in SQLite."""

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> EventGraphStore:
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return an open SQLite connection, creating it when needed."""
        return self.connect()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection lazily and ensure the schema exists."""
        if self._conn is not None:
            return self._conn

        if self.read_only:
            uri = f"file:{self.db_path}?mode=ro&cache=shared"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)

        conn.row_factory = sqlite3.Row
        self._apply_pragmas(conn)
        self._ensure_schema(conn)
        self._conn = conn
        return conn

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        """Configure pragmatic defaults for durability/performance."""
        conn.execute("PRAGMA foreign_keys = ON")
        if self.read_only:
            return
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Create tables/indexes and enforce a simple schema version."""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            for stmt in CREATE_STATEMENTS:
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return

        if version != SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported database schema version {version}; expected {SCHEMA_VERSION}")

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic transaction wrapper that rolls back automatically on failure."""
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def upsert_scenario(
        self,
        *,
        section: str,
        source_path: str,
        sheet_name: str,
        start_node_label: str,
        tags: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Insert or update a scenario entry and return its identifier."""
        now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        query = """
            INSERT INTO scenarios (section, source_path, sheet_name, start_node_label, ingested_at, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(section, source_path, sheet_name)
            DO UPDATE SET
                start_node_label=excluded.start_node_label,
                ingested_at=excluded.ingested_at,
                tags=excluded.tags
            RETURNING id
        """
        if conn is None:
            with self.transaction() as tx_conn:
                return self.upsert_scenario(
                    section=section,
                    source_path=source_path,
                    sheet_name=sheet_name,
                    start_node_label=start_node_label,
                    tags=tags,
                    conn=tx_conn,
                )

        row = conn.execute(
            query,
            (section, source_path, sheet_name, start_node_label, now, tags),
        ).fetchone()
        return int(row["id"])

    def fetch_scenario(
        self,
        *,
        scenario_id: int | None = None,
        section: str | None = None,
        source_path: str | None = None,
        sheet_name: str | None = None,
    ) -> ScenarioInfo:
        """
        Retrieve a single scenario row.

        Parameters
        ----------
        scenario_id : int | None, optional
            Numeric primary key. When provided, other filters are ignored.
        section : str | None, optional
            Section identifier; must be provided together with ``source_path`` and ``sheet_name`` when ``scenario_id`` is omitted.
        source_path : str | None, optional
            Path to the source Excel file; paired with ``section`` and ``sheet_name`` for unique lookup.
        sheet_name : str | None, optional
            Scenario worksheet name; required alongside ``section`` and ``source_path`` when ``scenario_id`` is omitted.

        Returns
        -------
        ScenarioInfo
            Scenario metadata mapped to strongly typed fields.

        Raises
        ------
        LookupError
            If no matching scenario could be found.
        ValueError
            When insufficient identifiers are provided.
        """
        conn = self.connect()
        if scenario_id is not None:
            row = conn.execute("SELECT * FROM scenarios WHERE id = ?", (scenario_id,)).fetchone()
        else:
            if not (section and source_path and sheet_name):
                raise ValueError("Provide scenario_id or (section, source_path, sheet_name)")
            row = conn.execute(
                """
                SELECT * FROM scenarios
                WHERE section = ? AND source_path = ? AND sheet_name = ?
                """,
                (section, source_path, sheet_name),
            ).fetchone()

        if row is None:
            raise LookupError("Scenario not found")
        return ScenarioInfo(
            id=int(row["id"]),
            section=row["section"],
            source_path=row["source_path"],
            sheet_name=row["sheet_name"],
            start_node_label=row["start_node_label"],
            ingested_at=row["ingested_at"],
            tags=row["tags"],
        )

    def list_scenarios(self, *, section: str | None = None) -> list[ScenarioInfo]:
        """Return all scenario rows, optionally filtered by section."""
        conn = self.connect()
        if section:
            rows = conn.execute(
                "SELECT * FROM scenarios WHERE section = ? ORDER BY source_path, sheet_name",
                (section,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM scenarios ORDER BY section, source_path, sheet_name").fetchall()
        return [
            ScenarioInfo(
                id=int(row["id"]),
                section=row["section"],
                source_path=row["source_path"],
                sheet_name=row["sheet_name"],
                start_node_label=row["start_node_label"],
                ingested_at=row["ingested_at"],
                tags=row["tags"],
            )
            for row in rows
        ]

    def delete_scenario_rows(
        self,
        scenario_id: int,
        tables: Sequence[str],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Remove all rows for ``scenario_id`` from the requested tables."""
        if conn is None:
            conn = self.connect()

        for table in tables:
            conn.execute(f"DELETE FROM {table} WHERE scenario_id = ?", (scenario_id,))

    def write_dataframe(
        self,
        table_name: str,
        df: pd.DataFrame,
        scenario_id: int,
        *,
        conn: sqlite3.Connection | None = None,
        replace: bool = True,
    ) -> None:
        """
        Persist a dataframe into one of the normalized tables for ``scenario_id``.

        Parameters
        ----------
        table_name : str
            Target table that must contain a ``scenario_id`` column.
        df : pd.DataFrame
            Dataframe to append.
        scenario_id : int
            Scenario foreign key.
        conn : sqlite3.Connection | None
            Optional open transaction connection; when omitted, the method creates
            a dedicated transaction.
        replace : bool
            When ``True`` existing rows for the scenario are removed before insert; when ``False`` data is appended.
        """
        if conn is None:
            conn = self.connect()

        if df is None or df.empty:
            if replace:
                self.delete_scenario_rows(scenario_id, [table_name], conn=conn)
            return

        if replace:
            conn.execute(f"DELETE FROM {table_name} WHERE scenario_id = ?", (scenario_id,))
        payload = df.copy()
        payload.insert(0, "scenario_id", scenario_id)
        payload = payload.astype(object).where(pd.notna(payload), None)
        columns = payload.columns.tolist()
        if not columns:
            return
        col_clause = ", ".join(f'"{col}"' for col in columns)
        placeholders = ", ".join(["?"] * len(columns))
        insert_sql = f"INSERT INTO {table_name} ({col_clause}) VALUES ({placeholders})"
        conn.executemany(insert_sql, payload.itertuples(index=False, name=None))

    def read_dataframe(self, table_name: str, scenario_id: int) -> pd.DataFrame:
        """Read rows linked to ``scenario_id`` and drop the ``scenario_id`` column."""
        query = f"SELECT * FROM {table_name} WHERE scenario_id = ?"
        df_sql = pd.read_sql_query(query, self.connect(), params=(scenario_id,))
        if "scenario_id" in df_sql.columns:
            df_sql = df_sql.drop(columns=["scenario_id"])
        return df_sql

    def write_graph(
        self,
        graph: nx.DiGraph,
        scenario_id: int,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist the provided ``networkx`` graph into the node/edge tables."""
        nodes_df, edges_df = serialize_graph(graph)
        if conn is None:
            with self.transaction() as tx_conn:
                self.write_dataframe("graph_nodes", nodes_df, scenario_id, conn=tx_conn)
                self.write_dataframe("graph_edges", edges_df, scenario_id, conn=tx_conn)
            return
        self.write_dataframe("graph_nodes", nodes_df, scenario_id, conn=conn)
        self.write_dataframe("graph_edges", edges_df, scenario_id, conn=conn)

    def read_graph(self, scenario_id: int) -> nx.DiGraph:
        """Load nodes and edges for ``scenario_id`` and rebuild the digraph."""
        nodes_df = self.read_dataframe("graph_nodes", scenario_id)
        edges_df = self.read_dataframe("graph_edges", scenario_id)
        return deserialize_graph(nodes_df, edges_df)


def serialize_graph(graph: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a :class:`networkx.DiGraph` into pandas dataframes."""
    node_rows = []
    for node_id, attrs in graph.nodes(data=True):
        faalpad_id, knoop_id = _split_node_id(node_id)
        node_rows.append(
            {
                "faalpad_id": faalpad_id,
                "knoop_id": knoop_id,
                "description": attrs.get("description"),
                "node_type": attrs.get("node_type"),
                "type_name": attrs.get("type_name"),
            }
        )

    edge_rows = []
    for source, target in graph.edges():
        s_faal, s_knoop = _split_node_id(source)
        t_faal, t_knoop = _split_node_id(target)
        edge_rows.append(
            {
                "source_faalpad": s_faal,
                "source_knoop": s_knoop,
                "target_faalpad": t_faal,
                "target_knoop": t_knoop,
            }
        )

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def deserialize_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Rebuild a :class:`networkx.DiGraph` from serialized node and edge frames."""
    graph = nx.DiGraph()
    for row in nodes_df.itertuples(index=False):
        node_id = (int(row.faalpad_id), int(row.knoop_id))
        graph.add_node(
            node_id,
            description=getattr(row, "description", None),
            node_type=getattr(row, "node_type", None),
            type_name=getattr(row, "type_name", None),
        )

    for row in edges_df.itertuples(index=False):
        source = (int(row.source_faalpad), int(row.source_knoop))
        target = (int(row.target_faalpad), int(row.target_knoop))
        graph.add_edge(source, target)

    return graph


def _split_node_id(node_id: object) -> tuple[int, int]:
    """Validate and split a tuple-based node identifier."""
    if isinstance(node_id, tuple) and len(node_id) == 2:
        return int(node_id[0]), int(node_id[1])
    raise TypeError(f"Node identifiers must be ``(faalpad_id, knoop_id)`` tuples; got {node_id!r}")
