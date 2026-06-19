import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from failure_paths.common.prob import beta_from_pf
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.curves import FragilityCurve, HazardCurve
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_diagnostics_1d

from failure_paths import ExcelEventGraph

range_water_levels = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# generate file list for all excel files in the example_input directory
input_folder = Path("scripts/example_input")
# file_list = list(input_folder.glob("*.xlsx"))
file_list = ["voorbeeld_GD_Overslag.xlsx"]


eeg = ExcelEventGraph.load("scripts/example_input/voorbeeld_GD_Overslag.xlsx", "scenario1")

# iterate over all water levels and files in the directory example_input
for file in file_list:
    fc_comb, fcs = eeg.get_failure_path_probabilities(water_levels=range_water_levels)
    for water_level in range_water_levels:
        eeg.plot(
            view=False,
            output_path=f"scripts/example_output/voorbeeld_DG_Overslag_waterlevel_{water_level}.png",
            water_level=water_level,
        )

    df_conditional = {}
    df_scenario = {}
    for fc in fcs:
        end_fc = fc.cumulative_probabilities.iloc[:, -1]
        endnode_name = eeg.graph.nodes[end_fc.name]["description"] + f" ({end_fc.name})"
        endnode_name = textwrap.fill(endnode_name, width=30)
        df_conditional[endnode_name] = end_fc.to_numpy()

        for nid in fc.path.nodes:
            node = eeg.graph.nodes[nid]
            if node["node_type"] == "scenario_node" and node["description"].endswith(": ja"):
                label = node["description"]
                if label not in df_scenario:
                    df_scenario[label] = fc.node_probabilities[nid].to_numpy()
    df_conditional = pd.DataFrame(df_conditional, index=range_water_levels)
    df_scenario = pd.DataFrame(df_scenario, index=range_water_levels)

    fig, axs = plt.subplots(ncols=2, figsize=(15, 6), dpi=100, sharey=True)
    df_conditional.plot(ax=axs[0], logy=True)
    axs[0].set_title("Conditional path fragility (excl. scenario weighting)")
    axs[0].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    axs[0].grid()
    df_scenario.plot(ax=axs[1], logy=True)
    fc_comb.plot(ax=axs[1], logy=True, color="black", linewidth=2.5, linestyle="--", label="Combined (total)")
    axs[1].legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    axs[1].set_title("Scenario probabilities and resulting total fragility")
    axs[1].grid()
    fig.tight_layout()
    plt.show()

    # Integrate the combined fragility curve against the water-level frequency
    # distribution (the "hfeq" sheet) to get one overall failure probability.
    hfeq = pd.read_excel("scripts/example_input/voorbeeld_GD_Overslag.xlsx", sheet_name="hfeq")
    dh = float(hfeq["h"].diff().iloc[1])
    exceedance_probs = np.clip(1.0 - np.cumsum(hfeq["PDF"].to_numpy()) * dh, 0.0, 1.0)

    hazard_curve = HazardCurve(hazard_levels=hfeq["h"].to_numpy(), exceedance_probs=exceedance_probs)
    fragility_curve = FragilityCurve(
        hazard_levels=fc_comb.index.to_numpy(),
        betas=beta_from_pf(fc_comb.to_numpy(), tail="upper"),
    )

    config = IntegrationConfig(hazard_curve=hazard_curve, fragility_curve=fragility_curve)
    result = ReliabilityIntegrator(config=config).run(collect_diagnostics_trace=True)
    print(result.summary())

    h_grid = np.linspace(
        min(hfeq["h"].min(), fc_comb.index.min()),
        max(hfeq["h"].max(), fc_comb.index.max()),
        300,
    )
    fig_curves, ax_curves = plt.subplots(figsize=(8, 5), dpi=100)
    ax_curves.plot(h_grid, hazard_curve.survival(h_grid), label="Hazard curve (exceedance frequency)")
    ax_curves.plot(h_grid, fragility_curve.cdf(h_grid), label="Fragility curve (combined path)")
    if result.design_point_physical is not None:
        ax_curves.axvline(
            result.design_point_physical.solicitation,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="design point",
        )
    ax_curves.set_yscale("log")
    ax_curves.set_xlabel("Water level")
    ax_curves.set_ylabel("Probability")
    ax_curves.set_title("Curves integrated by the reliability module")
    ax_curves.legend(fontsize="small")
    ax_curves.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    fig_curves.tight_layout()

    plot_integration_diagnostics_1d(result)

    plot_failure_histogram(result, hfeq["h"].to_numpy())

    plt.show()
