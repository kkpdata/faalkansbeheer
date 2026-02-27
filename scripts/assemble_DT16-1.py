import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openturns as ot
import pandas as pd
import tqdm.auto as tqdm
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.curves import FragilityCurve, HazardCurve
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_grid

from failure_paths import ExcelEventGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and integrate scenario fragility curves.")
    parser.add_argument(
        "--hr-path", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1/hr/WBI2017_Benedenrijn_16-1_v04")
    )
    parser.add_argument(
        "--hr-calname", default="ws"
    )  # map to calculation name in HR folder, as subfolder of "Berekeningen" (e.g. "ws", "hs", etc.)
    parser.add_argument("--dir-traject", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1/"))
    parser.add_argument("--output-folder", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1/output"))
    parser.add_argument("--scenario-name", default="scenario1")
    parser.add_argument("--wl-min", type=float, default=0.0)
    parser.add_argument("--wl-max", type=float, default=10.0)
    parser.add_argument("--wl-count", type=int, default=101)
    return parser.parse_args()


def integrate_and_plot(
    scen_name,
    hr_loc,
    fc_comb,
    hr_path,
    hr_calname,
    water_levels,
    fig_path,
):
    # Parse hfreq
    hfreq_path = hr_path / hr_loc / "Berekeningen" / hr_calname / "hfreq.txt"
    hfreq = pd.read_csv(hfreq_path, skiprows=1, header=None, sep="\\s+", index_col=0)[1]
    assert hfreq.max() < 1

    # Fragility curve -> beta
    pfs = fc_comb.to_numpy()
    fc_comb_beta = fc_comb.copy()
    fc_comb_beta.loc[:] = np.array(ot.Normal().computeQuantile(pfs, True)).reshape(pfs.shape)

    # Combine fragility curve with hfreq
    config = IntegrationConfig(
        fragility_curve=FragilityCurve(fc_comb_beta.index.to_numpy(), fc_comb_beta.to_numpy()),
        hazard_curve=HazardCurve(hfreq.index.to_numpy(), hfreq.to_numpy()),
        coarse_points=101,
        refine_factor=20,
        u_min=-10.0,
        u_max=10.0,
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run()

    # visualize integration grid
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5), dpi=100)
    plot_integration_grid(integrator, ax=axs[0])

    # visualize failure probability distribution over water levels
    _, _, hist_data = plot_failure_histogram(
        result, water_levels, ax=axs[1], conditional=False, solicitation_distribution=integrator.s_distribution
    )
    fig.savefig(fig_path / f"int_{scen_name}.png", bbox_inches="tight")
    plt.close("all")

    return result


def main() -> None:
    args = parse_args()
    hr_path = args.hr_path
    hr_calname = args.hr_calname
    dir_traject = args.dir_traject
    output_folder = args.output_folder
    scenario_name = args.scenario_name
    water_levels = np.linspace(args.wl_min, args.wl_max, args.wl_count)

    # Read all scenarios and structure them into sections
    sections = {}
    for p in dir_traject.rglob("*.xlsx", case_sensitive=False):
        if p.is_file() and not p.name.startswith("~$"):
            print(f"Loading scenario from {p}...")
            eeg = ExcelEventGraph.load(p, scenario_name)
            meta_row = eeg.metadata.df.iloc[0]
            section_name = f"{meta_row.TRAJECT_ID}_{meta_row.dijkvaknummer:03d}_{meta_row.Vaknaam}"
            if section_name not in sections:
                sections[section_name] = {}
            sections[section_name][meta_row.Ondergrondscenario] = (
                eeg,
                meta_row.ScenarioKans,
                meta_row.HR_locatie,
            )

    # Parse scenarios grouped by section
    df_result1 = {
        "Section": [],
        "HR_loc": [],
        "Scenario": [],
        "Scenario_weight": [],
        "Scenario_Pf": [],
        "Scenario_alpha_R": [],
        "Scenario_alpha_S": [],
        "Section_Pf": [],
        "Section_alpha_R": [],
        "Section_alpha_S": [],
    }
    sorted_sections = sorted(sections.items(), key=lambda item: item[0])
    for section_name, scenarios in tqdm.tqdm(sorted_sections, desc="Sections"):
        if len(scenarios) == 0:
            # no scenarios, continue to next section
            continue

        # Make a section folder
        write_path = output_folder / dir_traject.name / section_name
        fig_path = write_path / "figures"
        fig_path.mkdir(parents=True, exist_ok=True)

        fc_section = []
        hr_locs = []
        scen_probs = []
        df_plot_fc = {"water level": water_levels}
        for scen_name, (eeg, scen_prob, hr_loc) in tqdm.tqdm(scenarios.items(), leave=False, desc="Scenarios"):
            # Save tree plot
            eeg.plot(view=False, output_path=fig_path / f"tree_{scen_name}.png", water_level=3)

            # Get combined scenario fragility curve
            fc_comb, fcs = eeg.get_failure_path_probabilities(water_levels=water_levels)
            pfs = fc_comb.to_numpy()
            df_plot_fc[scen_name] = pfs

            # Add weighted scenario curve to section collection
            fc_section.append(pfs[:, np.newaxis] * scen_prob)
            hr_locs.append(hr_loc)
            scen_probs.append(scen_prob)

            # Integrate and plot scenario
            result = integrate_and_plot(
                scen_name,
                hr_loc,
                fc_comb,
                hr_path,
                hr_calname,
                water_levels,
                fig_path,
            )

            # Save scenario results
            df_result1["Section"].append(section_name)
            df_result1["HR_loc"].append(hr_loc)
            df_result1["Scenario"].append(scen_name)
            df_result1["Scenario_weight"].append(scen_prob)
            df_result1["Scenario_Pf"].append(result.pf)
            df_result1["Scenario_alpha_R"].append(result.alpha[0])
            df_result1["Scenario_alpha_S"].append(result.alpha[1])

        # Assert that only a single unique HR location is used for each scenario
        hr_locs = set(hr_locs)
        if len(hr_locs) != 1:
            raise ValueError("Multiple HR locations used for scenarios in a single section (can only be one)")

        # Assert that the scenario probabilities sum to 1
        if not np.isclose(sum(scen_probs), 1):
            raise ValueError("Scenario probabilities must sum to 1")

        # Get combined section fragility curve
        fc_section = np.hstack(fc_section)
        fc_section = np.array([math.fsum(row) for row in fc_section], dtype=float)
        df_plot_fc["combined"] = fc_section

        # Integrate combined section fragility curve
        fc_comb = fc_comb.copy()
        fc_comb[:] = fc_section
        result = integrate_and_plot(
            "combined",
            hr_loc,
            fc_comb,
            hr_path,
            hr_calname,
            water_levels,
            fig_path,
        )

        # Save section results
        for _ in scenarios:
            df_result1["Section_Pf"].append(result.pf)
            df_result1["Section_alpha_R"].append(result.alpha[0])
            df_result1["Section_alpha_S"].append(result.alpha[1])

        # Save fragility curve plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        df_plot_fc = pd.DataFrame(df_plot_fc)
        df_plot_fc.plot(ax=ax, x="water level", legend=True)
        ax.set_yscale("log")
        ax.grid()
        fig.savefig(fig_path / f"fc_{scen_name}.png", bbox_inches="tight")
        plt.close("all")

    # Write summary result
    df_result1 = pd.DataFrame(df_result1)
    df_result1.to_excel(output_folder / dir_traject.name / f"{dir_traject.name}_result.xlsx")


if __name__ == "__main__":
    main()
