"""Integration test for the assemble script using a Gumbel hazard curve.

Tests the integration of user-defined fragility curves with Gumbel-distributed hazards.
Tests the end-to-end workflow from Excel input to output verification. Results are compared against the probabilistic library implementation.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import probabilistic_library as pl
import pytest
from openpyxl import Workbook
from scipy.stats import gumbel_r

from failure_paths import ExcelEventGraph

matplotlib.use("Agg")


def test_assemble_fragility_curve_with_gumbel_hazard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    levels = [0.0, 3.0, 6.0]
    probabilities = [0.0001, 0.005, 0.1]
    gumbel_location, gumbel_scale = 2.0, 0.75

    trajectory = tmp_path / "trajectory"
    trajectory.mkdir()
    excel_path = trajectory / "scenario.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "scenario1"
    sheet.append(["metadata"])
    sheet.append(
        [
            "TRAJECT_ID",
            "M_VAN",
            "M_TOT",
            "Ondergrondscenario",
            "ScenarioKans",
            "LENGTE_VAK",
            "HR_locatie",
            "dijkvaknummer",
            "Vaknaam",
            "TYPE_WATERKERING",
        ]
    )
    sheet.append(["16-1", 0.0, 100.0, "scenario", 1.0, 100.0, "HR01", 1, "vak", "type"])
    sheet.append(["faalpadschema"])
    sheet.append(
        ["Faalpad_ID", "Overslag", "Indirect_mechanisme", "Initiele_gebeurtenis"]
        + [f"Vervolggebeurtenis_{i}" for i in range(1, 8)]
    )
    sheet.append([1, "nee", "nee", "falen"] + [None] * 7)
    sheet.append(["faalpadkansen"])
    sheet.append(["Faalpad_ID", "Knoop_ID", "h", "Pf_h", "Beta_h"])
    for level, probability in zip(levels, probabilities, strict=True):
        sheet.append([1, 1, level, probability, None])

    hazard = workbook.create_sheet("hazard")
    hazard.append(["h", "exceedance_probability"])
    for level in np.linspace(-0.5, 18.0, 371):
        hazard.append([float(level), float(gumbel_r.sf(level, loc=gumbel_location, scale=gumbel_scale))])
    workbook.save(excel_path)

    eeg = ExcelEventGraph.load(excel_path, "scenario1")
    assert [node for node, data in eeg.graph.nodes(data=True) if data["node_type"] == "failure_node"] == [(1, 1)]
    assert not eeg.freq_tables

    # assemble expects the sampled hazard in hfreq.txt, not in the workbook.
    hfreq_dir = tmp_path / "hr" / "HR01" / "Berekeningen" / "ws"
    hfreq_dir.mkdir(parents=True)
    with (hfreq_dir / "hfreq.txt").open("w") as hfreq:
        hfreq.write("water level exceedance probability\n")
        pd.read_excel(excel_path, sheet_name="hazard").to_csv(hfreq, sep=" ", index=False, header=False)

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import assemble

    monkeypatch.setattr(
        assemble,
        "parse_args",
        lambda: SimpleNamespace(
            hr_path=tmp_path / "hr",
            hr_calname="ws",
            dir_traject=trajectory,
            output_folder=tmp_path / "output",
            scenario_name="scenario1",
            wl_min=0.0,
            wl_max=6.0,
            wl_count=201,
            plot_beta=True,
            plot_tree=False,
            beta_inf_substitute=None,
            a_vak=0.5,
            delta_L=50.0,
            dijktraject="16-1",
            sqlite_db=None,
        ),
    )
    assemble.main()

    fragility = pl.FragilityCurve()
    for level, probability in zip(levels, probabilities, strict=True):
        value = pl.FragilityValue()
        value.x = level
        value.probability_of_failure = probability
        fragility.fragility_values.append(value)
    water_level = pl.Stochast()
    water_level.distribution = pl.DistributionType.gumbel
    water_level.scale = gumbel_scale
    water_level.shift = gumbel_location

    # Verify both packages interpret the curve between knots and in its extrapolated tails identically.
    for level, probability in zip(levels, probabilities, strict=True):
        assert eeg.graph_events.get_event_prob((1, 1), level) == pytest.approx(probability, rel=1e-10)
    for level in (-1.0, 1.5, 4.5, 7.0):
        assert eeg.graph_events.get_event_prob((1, 1), level) == pytest.approx(
            1.0 - fragility.get_cdf(level), rel=1e-10
        )
    for level in (0.0, 2.0, 6.0):
        assert water_level.get_cdf(level) == pytest.approx(
            gumbel_r.cdf(level, loc=gumbel_location, scale=gumbel_scale), rel=1e-10
        )

    reference = fragility.integrate(water_level)
    assert reference is not None
    assert reference.is_converged
    assert 0.0 < reference.probability_failure < 1.0
    results = pd.read_excel(tmp_path / "output" / trajectory.name / "trajectory_result.xlsx")
    assert len(results) == 1
    assert results.loc[0, "Scenario_Pf"] == pytest.approx(reference.probability_failure, rel=1e-4)
    assert results.loc[0, "Section_Pf"] == pytest.approx(reference.probability_failure, rel=1e-4)
