from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from failure_paths.eventgraph import ReservedPathId
from failure_paths.eventgraph.models import EventTable, MetadataTable, PathTable


def test_event_table_fills_pf_and_beta() -> None:
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [1, 1],
            "Knoop_ID": [1, 2],
            "h": [0.0, 0.0],
            "Pf_h": [0.01, np.nan],
            "Beta_h": [np.nan, 2.5],
        }
    )
    table = EventTable.from_dataframe(df_event)

    assert pytest.approx(table.df.loc[(1, 1), "Beta_h"]) == 2.3263478740408408
    assert pytest.approx(table.df.loc[(1, 2), "Pf_h"]) == 0.006209665325776159


def test_event_table_duplicate_events_raise() -> None:
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [1, 1],
            "Knoop_ID": [1, 1],
            "h": [0.0, 0.0],
            "Pf_h": [0.01, 0.02],
            "Beta_h": [np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError):
        EventTable.from_dataframe(df_event)


def test_event_table_non_monotone_fragility_raises() -> None:
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [1, 1, 1],
            "Knoop_ID": [1, 1, 1],
            "h": [0.0, 1.0, 2.0],
            "Pf_h": [0.1, 0.05, 0.2],
            "Beta_h": [np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        EventTable.from_dataframe(df_event)


def test_event_table_monotone_decreasing_fragility_allowed_for_special_faalpaden() -> None:
    indirect_id = ReservedPathId.INDIRECT_MECHANISM.value
    overslag_id = ReservedPathId.OVERTOPPING.value
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [indirect_id, indirect_id, overslag_id, overslag_id],
            "Knoop_ID": [1, 1, 1, 1],
            "h": [0.0, 1.0, 0.0, 1.0],
            "Pf_h": [0.2, 0.1, 0.3, 0.2],
            "Beta_h": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    table = EventTable.from_dataframe(df_event)

    assert len(table.df) == 4


def test_event_table_monotone_increasing_fragility_allowed_for_special_faalpaden() -> None:
    indirect_id = ReservedPathId.INDIRECT_MECHANISM.value
    overslag_id = ReservedPathId.OVERTOPPING.value
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [indirect_id, indirect_id, overslag_id, overslag_id],
            "Knoop_ID": [1, 1, 1, 1],
            "h": [0.0, 1.0, 0.0, 1.0],
            "Pf_h": [0.1, 0.2, 0.2, 0.3],
            "Beta_h": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    table = EventTable.from_dataframe(df_event)

    assert len(table.df) == 4


def test_event_table_mixed_direction_fragility_raises_for_special_faalpaden() -> None:
    indirect_id = ReservedPathId.INDIRECT_MECHANISM.value
    df_event = pd.DataFrame(
        {
            "Faalpad_ID": [indirect_id, indirect_id, indirect_id],
            "Knoop_ID": [1, 1, 1],
            "h": [0.0, 1.0, 2.0],
            "Pf_h": [0.2, 0.1, 0.15],
            "Beta_h": [np.nan, np.nan, np.nan],
        }
    )

    special_ids_label = f"{ReservedPathId.INDIRECT_MECHANISM.value}/{ReservedPathId.OVERTOPPING.value}"
    with pytest.raises(ValueError, match=rf"special Faalpad_ID {special_ids_label} must be monotone"):
        EventTable.from_dataframe(df_event)


def test_path_table_normalizes_inputs() -> None:
    df_path = pd.DataFrame(
        {
            "Faalpad_ID": [1],
            "Overslag": ["JA"],
            "Indirect_mechanisme": [" Nee "],
            "Initiele_gebeurtenis": ["init"],
            "Vervolggebeurtenis_1": ["e1"],
            "Vervolggebeurtenis_2": ["e2"],
            "Vervolggebeurtenis_3": [pd.NA],
            "Vervolggebeurtenis_4": [pd.NA],
            "Vervolggebeurtenis_5": [pd.NA],
            "Vervolggebeurtenis_6": [pd.NA],
            "Vervolggebeurtenis_7": [pd.NA],
        }
    )
    table = PathTable.from_dataframe(df_path)
    normalized = table.df.reset_index()
    assert normalized.loc[0, "Overslag"] == "ja"
    assert normalized.loc[0, "Indirect_mechanisme"] == "nee"


def test_metadata_table_missing_required_columns_raises() -> None:
    df_metadata = pd.DataFrame(
        {
            "Ondergrondscenario": ["Scenario"],
            "LENGTE_VAK": [1.0],
            "HR_locatie": ["HR01"],
        }
    )
    with pytest.raises(ValueError):
        MetadataTable.from_dataframe(df_metadata)
