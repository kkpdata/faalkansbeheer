from __future__ import annotations

import networkx as nx
import pandas as pd
from failure_paths.eventgraph import ReservedPathId
from failure_paths.eventgraph.sqlite_store import deserialize_graph, serialize_graph


def build_sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    start_id = ReservedPathId.START_NODE.value
    overslag_id = ReservedPathId.OVERTOPPING.value
    graph.add_node(
        (start_id, 0),
        description="start node",
        node_type="start_node",
        type_name=None,
    )
    graph.add_node(
        (overslag_id, 1),
        description="scenario overslag",
        node_type="scenario_node",
        type_name="overslag",
    )
    graph.add_node(
        (1, 1),
        description="event one",
        node_type="event_node",
        type_name=None,
    )
    graph.add_node(
        (1, 2),
        description="failure node",
        node_type="failure_node",
        type_name=None,
    )
    graph.add_edge((start_id, 0), (overslag_id, 1))
    graph.add_edge((overslag_id, 1), (1, 1))
    graph.add_edge((1, 1), (1, 2))
    return graph


def test_serialize_graph_returns_expected_frames() -> None:
    graph = build_sample_graph()
    nodes_df, edges_df = serialize_graph(graph)

    assert set(nodes_df.columns) == {"faalpad_id", "knoop_id", "description", "node_type", "type_name"}
    assert len(nodes_df) == 4
    pd.testing.assert_series_equal(
        nodes_df["faalpad_id"],
        pd.Series(
            [
                ReservedPathId.START_NODE.value,
                ReservedPathId.OVERTOPPING.value,
                1,
                1,
            ],
            name="faalpad_id",
        ),
        check_names=True,
        check_dtype=False,
    )

    assert set(edges_df.columns) == {
        "source_faalpad",
        "source_knoop",
        "target_faalpad",
        "target_knoop",
    }
    assert len(edges_df) == 3


def test_deserialize_graph_round_trip() -> None:
    graph = build_sample_graph()
    nodes_df, edges_df = serialize_graph(graph)

    restored = deserialize_graph(nodes_df, edges_df)

    assert set(restored.nodes) == set(graph.nodes)
    assert set(restored.edges) == set(graph.edges)
    for node_id in graph.nodes:
        restored_attrs = dict(restored.nodes[node_id])
        original_attrs = dict(graph.nodes[node_id])
        if pd.isna(restored_attrs.get("type_name")) and original_attrs.get("type_name") is None:
            restored_attrs["type_name"] = None
        assert restored_attrs == original_attrs
