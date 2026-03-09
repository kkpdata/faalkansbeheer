from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from failure_paths.eventgraph import EventGraph
from failure_paths.eventgraph.models import EventTable, MetadataTable


class DummyEventGraph(EventGraph):
    @classmethod
    def load(cls):
        raise NotImplementedError


def build_dummy_graph() -> DummyEventGraph:
    metadata = MetadataTable.from_dataframe(
        pd.DataFrame(
            {
                "TRAJECT_ID": ["16-X"],
                "M_VAN": [0.0],
                "M_TOT": [100.0],
                "Ondergrondscenario": ["Scenario"],
                "ScenarioKans": [0.2],
                "LENGTE_VAK": [1.0],
                "HR_locatie": ["HR"],
                "dijkvaknummer": [1],
                "Vaknaam": ["Vak"],
                "TYPE_WATERKERING": ["type"],
            }
        )
    )
    events = EventTable.from_dataframe(
        pd.DataFrame(
            {
                "Faalpad_ID": [1, 1],
                "Knoop_ID": [1, 1],
                "h": [0.0, 1.0],
                "Pf_h": [0.01, 0.02],
                "Beta_h": [2.3263, 2.0537],
            }
        )
    )
    graph = nx.DiGraph()
    graph.add_node((-3, 0), description="start node", node_type="start_node", type_name=None)
    graph.add_node((1, 1), description="failure node label", node_type="failure_node", type_name=None)
    graph.add_edge((-3, 0), (1, 1))
    return DummyEventGraph(metadata=metadata, graph=graph, graph_events=events)


def build_branching_graph() -> DummyEventGraph:
    metadata = MetadataTable.from_dataframe(
        pd.DataFrame(
            {
                "TRAJECT_ID": ["16-X"],
                "M_VAN": [0.0],
                "M_TOT": [100.0],
                "Ondergrondscenario": ["Scenario"],
                "ScenarioKans": [0.2],
                "LENGTE_VAK": [1.0],
                "HR_locatie": ["HR"],
                "dijkvaknummer": [1],
                "Vaknaam": ["Vak"],
                "TYPE_WATERKERING": ["type"],
            }
        )
    )
    rows = []
    specs = [
        (1, 10, 0.01, 0.02),
        (1, 99, 0.015, 0.025),
        (2, 10, 0.02, 0.03),
        (2, 99, 0.03, 0.04),
    ]
    for faalpad_id, knoop_id, pf_low, pf_high in specs:
        rows.append(
            {
                "Faalpad_ID": faalpad_id,
                "Knoop_ID": knoop_id,
                "h": 0.0,
                "Pf_h": pf_low,
                "Beta_h": 2.3263,
            }
        )
        rows.append(
            {
                "Faalpad_ID": faalpad_id,
                "Knoop_ID": knoop_id,
                "h": 1.0,
                "Pf_h": pf_high,
                "Beta_h": 2.0537,
            }
        )
    events = EventTable.from_dataframe(pd.DataFrame(rows))
    graph = nx.DiGraph()
    root_a = (-3, 0)
    root_b = (-3, 1)
    event_a = (1, 10)
    event_b = (2, 10)
    failure_a = (1, 99)
    failure_b = (2, 99)

    graph.add_node(root_a, description="start a", node_type="start_node", type_name=None)
    graph.add_node(root_b, description="start b", node_type="start_node", type_name=None)
    graph.add_node(event_a, description="event a", node_type="event_node", type_name=None)
    graph.add_node(event_b, description="event b", node_type="event_node", type_name=None)
    graph.add_node(failure_a, description="failure a", node_type="failure_node", type_name=None)
    graph.add_node(failure_b, description="failure b", node_type="failure_node", type_name=None)

    graph.add_edges_from(
        [
            (root_a, event_a),
            (event_a, failure_a),
            (root_a, event_b),
            (root_b, event_b),
            (event_b, failure_b),
        ]
    )

    return DummyEventGraph(metadata=metadata, graph=graph, graph_events=events)


def test_format_label_wraps_text() -> None:
    text = "this is a long description"
    wrapped = EventGraph._format_label(text, width=5)
    assert "\n" in wrapped
    assert "this" in wrapped


def test_build_graphviz_includes_pf_annotation(tmp_path) -> None:
    dummy = build_dummy_graph()
    dot = dummy._build_graphviz(
        wrap_width=30,
        water_level=0.0,
        graph_attr=None,
        node_attr=None,
        edge_attr=None,
        event_attr=None,
    )

    body = "\n".join(dot.body)
    assert "pf=" in body
    assert "failure node label" in body


def test_get_failure_paths_defaults_to_roots() -> None:
    dummy = build_branching_graph()
    paths = dummy.get_failure_paths()
    expected = {
        ((-3, 0), (1, 10), (1, 99)),
        ((-3, 0), (2, 10), (2, 99)),
        ((-3, 1), (2, 10), (2, 99)),
    }
    assert {tuple(path.nodes) for path in paths} == expected


def test_get_failure_paths_honors_custom_start_nodes() -> None:
    dummy = build_branching_graph()
    paths = dummy.get_failure_paths(start_nodes=[(-3, 1)])
    expected = {((-3, 1), (2, 10), (2, 99))}
    assert {tuple(path.nodes) for path in paths} == expected


def test_get_failure_path_probabilities_single_level() -> None:
    dummy = build_branching_graph()
    _, results = dummy.get_failure_path_probabilities(0.0)
    assert len(results) == 3
    mapping = {tuple(entry.path.nodes): entry for entry in results}
    target_path = ((-3, 0), (1, 10), (1, 99))
    target = mapping[target_path]
    np.testing.assert_allclose(target.water_levels, np.array([0.0]))
    matrix = target.node_probabilities
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (1, len(target_path))
    assert list(matrix.columns) == list(target_path)
    np.testing.assert_allclose(matrix.index.values, np.array([0.0]))
    start_val = matrix.at[0.0, (-3, 0)]
    assert start_val == pytest.approx(1.0)
    event_val = dummy.graph_events.get_event_prob((1, 10), 0.0)
    assert matrix.at[0.0, (1, 10)] == pytest.approx(event_val)
    fail_val = dummy.graph_events.get_event_prob((1, 99), 0.0)
    assert matrix.at[0.0, (1, 99)] == pytest.approx(fail_val)
    cum_df = target.cumulative_probabilities
    assert isinstance(cum_df, pd.DataFrame)
    assert cum_df.shape == matrix.shape
    assert list(cum_df.columns) == list(target_path)
    assert np.isnan(cum_df.at[0.0, (-3, 0)])
    assert cum_df.at[0.0, (1, 10)] == pytest.approx(event_val)
    assert cum_df.at[0.0, (1, 99)] == pytest.approx(event_val * fail_val)


def test_get_failure_path_probabilities_multiple_levels() -> None:
    dummy = build_branching_graph()
    levels = [0.0, 1.0]
    _, results = dummy.get_failure_path_probabilities(levels, start_nodes=[(-3, 0)])
    assert len(results) == 2
    mapping = {tuple(entry.path.nodes): entry for entry in results}
    target_path = ((-3, 0), (2, 10), (2, 99))
    target = mapping[target_path]
    np.testing.assert_allclose(target.water_levels, np.array(levels))
    matrix = target.node_probabilities
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (len(levels), len(target_path))
    assert list(matrix.columns) == list(target_path)
    np.testing.assert_allclose(matrix.index.values, np.array(levels))
    np.testing.assert_allclose(
        matrix[(-3, 0)].to_numpy(),
        np.array([1.0, 1.0]),
    )
    expected = dummy.graph_events.get_event_probs((2, 10), levels)
    np.testing.assert_allclose(matrix[(2, 10)].to_numpy(), expected)
    expected_failure = dummy.graph_events.get_event_probs((2, 99), levels)
    np.testing.assert_allclose(matrix[(2, 99)].to_numpy(), expected_failure)
    cum_df = target.cumulative_probabilities
    assert isinstance(cum_df, pd.DataFrame)
    np.testing.assert_allclose(
        cum_df.to_numpy()[:, 1:],
        np.cumprod(matrix.to_numpy()[:, 1:], axis=1),
    )


def test_get_failure_path_probabilities_near_vertical_curve_end_to_end() -> None:
    dummy = build_dummy_graph()
    h_low = 3.0
    h_high = 3.000001
    h_mid = (h_low + h_high) / 2.0
    dummy.graph_events = EventTable.from_dataframe(
        pd.DataFrame(
            {
                "Faalpad_ID": [1, 1],
                "Knoop_ID": [1, 1],
                "h": [h_low, h_high],
                "Pf_h": [0.0, 1.0],
                "Beta_h": [np.nan, np.nan],
            }
        )
    )

    levels = [h_low, h_mid, h_high]
    fc, results = dummy.get_failure_path_probabilities(levels)

    np.testing.assert_allclose(fc.index.values, np.array(levels))
    np.testing.assert_allclose(fc.to_numpy(), np.array([1.0e-300, 0.5, 1.0]))
    matrix = results[0].node_probabilities
    assert matrix.at[h_low, (1, 1)] == pytest.approx(0.0)
    assert matrix.at[h_mid, (1, 1)] == pytest.approx(0.5)
    assert matrix.at[h_high, (1, 1)] == pytest.approx(1.0)
