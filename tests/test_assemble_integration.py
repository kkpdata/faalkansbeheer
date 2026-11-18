"""Integration test for the assemble script using normal distributions.

Tests the integration of a normal distributed fragility curves with normally distributed hazards.
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
from scipy.stats import norm

from failure_paths import ExcelEventGraph

matplotlib.use("Agg")


def _limit_state(resistance: float, water_level: float) -> float:
    return resistance - water_level


def test_assemble_matches_probabilistic_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Linear interpolation in beta-space represents these normal distributions exactly.
    resistance_mean, resistance_sd = 3.0, 1.0
    water_mean, water_sd = 1.5, 0.7
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
    for level in (0.0, 4.0):
        sheet.append([1, 1, level, norm.cdf(level, loc=resistance_mean, scale=resistance_sd), None])

    hazard = workbook.create_sheet("hazard")
    hazard.append(["h", "exceedance_probability"])
    for level in (0.0, 3.0):
        hazard.append([level, norm.sf(level, loc=water_mean, scale=water_sd)])
    workbook.save(excel_path)

    eeg = ExcelEventGraph.load(excel_path, "scenario1")
    assert [node for node, data in eeg.graph.nodes(data=True) if data["node_type"] == "failure_node"] == [(1, 1)]
    assert not eeg.freq_tables

    # assemble reads the hazard curve from hfreq.txt, not directly from the workbook.
    hfreq_dir = tmp_path / "hr" / "HR01" / "Berekeningen" / "ws"
    hfreq_dir.mkdir(parents=True)
    hazard_data = pd.read_excel(excel_path, sheet_name="hazard")
    with (hfreq_dir / "hfreq.txt").open("w") as hfreq:
        hfreq.write("water level exceedance probability\n")
        hazard_data.to_csv(hfreq, sep=" ", index=False, header=False)

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
            wl_max=4.0,
            wl_count=101,
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

    project = pl.ReliabilityProject()
    project.model = _limit_state
    for variable, mean, deviation in zip(
        project.variables, (resistance_mean, water_mean), (resistance_sd, water_sd), strict=True
    ):
        variable.distribution = pl.DistributionType.normal
        variable.mean = mean
        variable.deviation = deviation
    project.settings.reliability_method = pl.ReliabilityMethod.form
    assert project.is_valid()
    project.run()
    reference = project.design_point
    assert reference is not None
    assert reference.is_converged
    assert reference.probability_failure == pytest.approx(
        norm.cdf((water_mean - resistance_mean) / np.hypot(resistance_sd, water_sd)), rel=1e-5
    ) # analytical solution for the probability of failure

    results = pd.read_excel(tmp_path / "output" / trajectory.name / "trajectory_result.xlsx")
    assert len(results) == 1
    assert results.loc[0, "Scenario_Pf"] == pytest.approx(reference.probability_failure, rel=1e-5)

