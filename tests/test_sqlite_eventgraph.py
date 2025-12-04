from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from failure_paths.models import EventTable, MetadataTable
from failure_paths.sqlite_store import EventGraphStore, ScenarioInfo

from failure_paths import SqliteEventGraph


def build_metadata_table() -> MetadataTable:
    df_metadata = pd.DataFrame(
        {
            "Ondergrondscenario": ["ScenarioA"],
            "ScenarioKans": [0.1],
            "LENGTE_VAK": [1.23],
            "HR_locatie": ["HR01"],
            "dijkvaknummer": [42],
            "Vaknaam": ["Vak 42"],
            "TYPE_WATERKERING": ["type"],
        }
    )
    return MetadataTable.from_dataframe(df_metadata)


def build_event_table() -> EventTable:
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [1, 1, 1, 1],
            "Knoop_ID": [1, 1, 2, 2],
            "h": [0.0, 1.0, 0.0, 1.0],
            "Pf_h": [0.01, 0.02, 0.02, 0.03],
            "Beta_h": [2.3263, 2.0537, 2.0537, 1.8808],
        }
    )
    return EventTable.from_dataframe(df_event)


def build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node((-3, 0), description="start", node_type="start_node", type_name=None)
    graph.add_node((1, 1), description="event1", node_type="event_node", type_name=None)
    graph.add_node((1, 2), description="failure", node_type="failure_node", type_name=None)
    graph.add_edge((-3, 0), (1, 1))
    graph.add_edge((1, 1), (1, 2))
    return graph


def persist_sample_scenario(
    db_path: Path,
) -> tuple[int, MetadataTable, EventTable, nx.DiGraph]:
    metadata = build_metadata_table()
    events = build_event_table()
    graph = build_graph()

    with EventGraphStore(db_path) as store:
        with store.transaction() as conn:
            scenario_id = store.upsert_scenario(
                section="A",
                source_path="test.xlsx",
                sheet_name="scenario1",
                start_node_label="test.xlsx: scenario1",
                tags=None,
                conn=conn,
            )
            store.write_dataframe("metadata", metadata.to_records(), scenario_id, conn=conn)
            store.write_dataframe("events", events.to_required_records(), scenario_id, conn=conn)
            store.write_graph(graph, scenario_id, conn=conn)

    return scenario_id, metadata, events, graph


def test_sqlite_eventgraph_loads_from_database(tmp_path: Path) -> None:
    db_path = tmp_path / "scenarios.db"
    scenario_id, metadata, events, graph = persist_sample_scenario(db_path)

    loaded = SqliteEventGraph.load(db_path, scenario_id=scenario_id)

    assert loaded.scenario_id == scenario_id
    pd.testing.assert_frame_equal(
        loaded.metadata.to_dataframe().reset_index(drop=True),
        metadata.to_dataframe().reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        loaded.graph_events.to_dataframe().reset_index(drop=True),
        events.to_dataframe().reset_index(drop=True),
    )
    assert set(loaded.graph.nodes) == set(graph.nodes)
    assert set(loaded.graph.edges) == set(graph.edges)

    scenarios = SqliteEventGraph.available_scenarios(db_path)
    assert len(scenarios) == 1
    assert scenarios[0].section == "A"


def test_eventgraph_store_rolls_back_transactions(tmp_path: Path) -> None:
    db_path = tmp_path / "rollback.db"
    metadata = build_metadata_table()
    events = build_event_table()

    with EventGraphStore(db_path) as store:
        with pytest.raises(RuntimeError):
            with store.transaction() as conn:
                scenario_id = store.upsert_scenario(
                    section="A",
                    source_path="rollback.xlsx",
                    sheet_name="scenario1",
                    start_node_label="rollback: scenario1",
                    tags=None,
                    conn=conn,
                )
                store.write_dataframe("metadata", metadata.to_records(), scenario_id, conn=conn)
                store.write_dataframe("events", events.to_required_records(), scenario_id, conn=conn)
                raise RuntimeError("abort transaction")

        assert store.list_scenarios() == []


def test_scenario_lookup_variants(tmp_path: Path) -> None:
    db_path = tmp_path / "lookup.db"
    scenario_id, *_ = persist_sample_scenario(db_path)

    with EventGraphStore(db_path) as store:
        scenario = store.fetch_scenario(scenario_id=scenario_id)
        assert scenario.source_path.endswith("test.xlsx")

        scenario_alt = store.fetch_scenario(section="A", source_path="test.xlsx", sheet_name="scenario1")
        assert scenario_alt.id == scenario_id

        scenarios = store.list_scenarios()
        assert len(scenarios) == 1
        assert isinstance(scenarios[0], ScenarioInfo)

        with pytest.raises(LookupError):
            store.fetch_scenario(scenario_id=scenario_id + 1)


def test_sqlite_eventgraph_preserves_probabilities_and_attributes(tmp_path: Path) -> None:
    db_path = tmp_path / "prob.db"
    scenario_id, _metadata, events, graph = persist_sample_scenario(db_path)

    loaded = SqliteEventGraph.load(db_path, scenario_id=scenario_id)

    prob = loaded.graph_events.get_event_prob((1, 1), h=0.0)
    assert prob == pytest.approx(events.get_event_prob((1, 1), 0.0))

    for node_id in graph.nodes:
        assert loaded.graph.nodes[node_id] == graph.nodes[node_id]
