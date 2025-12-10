from __future__ import annotations

import networkx as nx
import pandas as pd
from failure_paths.eventgraph.sqlite_store import deserialize_graph, serialize_graph


def build_sample_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(
        (-3, 0),
        description="start node",
        node_type="start_node",
        type_name=None,
    )
    graph.add_node(
        (-2, 1),
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
    graph.add_edge((-3, 0), (-2, 1))
    graph.add_edge((-2, 1), (1, 1))
    graph.add_edge((1, 1), (1, 2))
    return graph


def test_serialize_graph_returns_expected_frames() -> None:
    graph = build_sample_graph()
    nodes_df, edges_df = serialize_graph(graph)

    assert set(nodes_df.columns) == {"faalpad_id", "knoop_id", "description", "node_type", "type_name"}
    assert len(nodes_df) == 4
    pd.testing.assert_series_equal(
        nodes_df["faalpad_id"],
        pd.Series([-3, -2, 1, 1], name="faalpad_id"),
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
        assert restored.nodes[node_id] == graph.nodes[node_id]
