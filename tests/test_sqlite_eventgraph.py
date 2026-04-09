from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from failure_paths.eventgraph import EventGraphStore, ScenarioInfo
from failure_paths.eventgraph.models import EventTable, FrequencyTable, MetadataTable

from failure_paths import SqliteEventGraph


def build_metadata_table() -> MetadataTable:
    df_metadata = pd.DataFrame(
        {
            "TRAJECT_ID": ["16-X"],
            "M_VAN": [0.0],
            "M_TOT": [100.0],
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


def build_graph_alt() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node((-3, 0), description="start", node_type="start_node", type_name=None)
    graph.add_node((1, 1), description="event2", node_type="event_node", type_name=None)
    graph.add_node((1, 2), description="failure2", node_type="failure_node", type_name=None)
    graph.add_edge((-3, 0), (1, 1))
    graph.add_edge((1, 1), (1, 2))
    return graph


def build_frequency_tables() -> dict[str, FrequencyTable]:
    return {
        "overslag": FrequencyTable.from_dataframe(
            pd.DataFrame(
                {
                    "h": [0.0, 1.0],
                    "Pf_h": [0.5, 0.6],
                }
            )
        ),
        "custom": FrequencyTable.from_dataframe(
            pd.DataFrame(
                {
                    "h": [0.0, 1.0],
                    "Pf_h": [0.2, 0.3],
                }
            )
        ),
    }


def persist_sample_scenario(
    db_path: Path,
) -> tuple[int, MetadataTable, EventTable, nx.DiGraph, dict[str, FrequencyTable]]:
    metadata = build_metadata_table()
    events = build_event_table()
    graph = build_graph()
    freq_tables = build_frequency_tables()

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
            store.write_frequency_tables(freq_tables, scenario_id, conn=conn)

    return scenario_id, metadata, events, graph, freq_tables


def test_sqlite_eventgraph_loads_from_database(tmp_path: Path) -> None:
    db_path = tmp_path / "scenarios.db"
    scenario_id, metadata, events, graph, freq_tables = persist_sample_scenario(db_path)

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
    assert list(loaded.freq_tables.keys()) == list(freq_tables.keys())
    for key in freq_tables:
        pd.testing.assert_frame_equal(
            loaded.freq_tables[key].to_dataframe().reset_index(drop=True),
            freq_tables[key].to_dataframe().reset_index(drop=True),
        )

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


def test_store_and_sqlite_eventgraph_can_list_sections_and_node_names(tmp_path: Path) -> None:
    db_path = tmp_path / "indexing.db"
    metadata = build_metadata_table()
    events = build_event_table()
    graph_a = build_graph()
    graph_b = build_graph_alt()

    with EventGraphStore(db_path) as store:
        with store.transaction() as conn:
            scenario_id_a = store.upsert_scenario(
                section="A",
                source_path="a.xlsx",
                sheet_name="scenario1",
                start_node_label="a.xlsx: scenario1",
                tags=None,
                conn=conn,
            )
            store.write_dataframe("metadata", metadata.to_records(), scenario_id_a, conn=conn)
            store.write_dataframe("events", events.to_required_records(), scenario_id_a, conn=conn)
            store.write_graph(graph_a, scenario_id_a, conn=conn)

            scenario_id_b = store.upsert_scenario(
                section="B",
                source_path="b.xlsx",
                sheet_name="scenario1",
                start_node_label="b.xlsx: scenario1",
                tags=None,
                conn=conn,
            )
            store.write_dataframe("metadata", metadata.to_records(), scenario_id_b, conn=conn)
            store.write_dataframe("events", events.to_required_records(), scenario_id_b, conn=conn)
            store.write_graph(graph_b, scenario_id_b, conn=conn)

        assert store.list_sections() == ["A", "B"]
        assert store.list_node_names() == ["event1", "event2", "failure", "failure2", "start"]
        assert store.list_node_names(section="A") == ["event1", "failure", "start"]
        assert store.list_node_names(section="B", node_types=["event_node"]) == ["event2"]
        assert store.list_node_names(section="B", node_types=[]) == []

    assert SqliteEventGraph.available_sections(db_path) == ["A", "B"]
    assert SqliteEventGraph.available_node_names(db_path, section="A", node_types=["failure_node"]) == ["failure"]


def test_sqlite_eventgraph_preserves_probabilities_and_attributes(tmp_path: Path) -> None:
    db_path = tmp_path / "prob.db"
    scenario_id, _metadata, events, graph, _freq_tables = persist_sample_scenario(db_path)

    loaded = SqliteEventGraph.load(db_path, scenario_id=scenario_id)

    prob = loaded.graph_events.get_event_prob((1, 1), h=0.0)
    assert prob == pytest.approx(events.get_event_prob((1, 1), 0.0))

    for node_id in graph.nodes:
        assert loaded.graph.nodes[node_id] == graph.nodes[node_id]


def test_sqlite_eventgraph_frequency_tables_preserve_order_and_duplicate_h(tmp_path: Path) -> None:
    db_path = tmp_path / "freq.db"
    metadata = build_metadata_table()
    events = build_event_table()
    graph = build_graph()
    freq_tables = {
        "overslag": FrequencyTable.from_dataframe(
            pd.DataFrame(
                {
                    "h": [0.0, 0.5, 0.5, 1.0],
                    "Pf_h": [0.4, 0.6, 0.65, 0.7],
                }
            )
        ),
        "custom": FrequencyTable.from_dataframe(
            pd.DataFrame(
                {
                    "h": [0.0, 1.0],
                    "Pf_h": [0.25, 0.35],
                }
            )
        ),
    }

    with EventGraphStore(db_path) as store:
        with store.transaction() as conn:
            scenario_id = store.upsert_scenario(
                section="A",
                source_path="freq.xlsx",
                sheet_name="scenario1",
                start_node_label="freq.xlsx: scenario1",
                tags=None,
                conn=conn,
            )
            store.write_dataframe("metadata", metadata.to_records(), scenario_id, conn=conn)
            store.write_dataframe("events", events.to_required_records(), scenario_id, conn=conn)
            store.write_graph(graph, scenario_id, conn=conn)
            store.write_frequency_tables(freq_tables, scenario_id, conn=conn)

    loaded = SqliteEventGraph.load(db_path, scenario_id=scenario_id)
    assert list(loaded.freq_tables.keys()) == ["overslag", "custom"]
    for key, original in freq_tables.items():
        pd.testing.assert_frame_equal(
            loaded.freq_tables[key].to_dataframe().reset_index(drop=True),
            original.to_dataframe().reset_index(drop=True),
        )
