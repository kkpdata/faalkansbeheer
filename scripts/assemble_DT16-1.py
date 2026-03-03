import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from failure_paths.common.interp import LinearInterpolator
from failure_paths.common.prob import beta_from_pf, pf_from_beta
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.curves import FragilityCurve, HazardCurve
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_grid

from failure_paths import ExcelEventGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and integrate scenario fragility curves.")
    parser.add_argument(
        "--hr-path", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1_hr/WBI2017_Benedenrijn_16-1_v04")
    )
    parser.add_argument("--hr-calname", default="ws")
    parser.add_argument("--dir-traject", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1"))
    parser.add_argument("--output-folder", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-1_output"))
    parser.add_argument("--scenario-name", default="scenario1")
    parser.add_argument("--wl-min", type=float, default=0.0)
    parser.add_argument("--wl-max", type=float, default=10.0)
    parser.add_argument("--wl-count", type=int, default=101)
    parser.add_argument("--plot-beta", type=bool, default=False)
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
    fc_comb_beta = fc_comb.copy()
    fc_comb_beta.loc[:] = beta_from_pf(fc_comb.to_numpy())

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
    if scen_name == "combined":
        fig.savefig(fig_path / f"int_section_{scen_name}.png", bbox_inches="tight")
    else:
        fig.savefig(fig_path / f"int_scenario_{scen_name}.png", bbox_inches="tight")
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
    plot_beta = args.plot_beta

    # Read all scenarios and structure them into sections
    sections = {}
    meta_cols = []
    for p in dir_traject.rglob("*.xlsx", case_sensitive=False):
        if p.is_file() and not p.name.startswith("~$"):
            try:
                print(f"Loading '{p}'...")
                eeg = ExcelEventGraph.load(p, scenario_name)
            except Exception as e:
                print(f"Error while loading '{p}':")
                print(e)
                continue
            meta_row = eeg.metadata.df.iloc[0]
            section_name = f"{meta_row.TRAJECT_ID}_{meta_row.dijkvaknummer:03d}_{meta_row.Vaknaam}"
            if section_name not in sections:
                sections[section_name] = {}
            sections[section_name][meta_row.Ondergrondscenario] = (
                eeg,
                meta_row.ScenarioKans,
                meta_row.HR_locatie,
            )
            meta_cols = pd.unique(np.array(meta_cols + eeg.metadata.df.columns.tolist())).tolist()

    # Parse scenarios grouped by section
    df_result1 = {m: [] for m in meta_cols}
    df_result1["Section"] = []
    df_result1["Scenario_weight"] = []
    df_result1["Scenario_Pf"] = []
    df_result1["Scenario_alpha_R"] = []
    df_result1["Scenario_alpha_S"] = []
    df_result1["Section_Pf"] = []
    df_result1["Section_alpha_R"] = []
    df_result1["Section_alpha_S"] = []

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
            eeg.plot(view=False, output_path=fig_path / f"tree_scenario_{scen_name}.png", water_level=3)

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

            # Gather scenario curves (if present)
            df_fc_paths = {}
            for table_name, table in eeg.freq_tables.items():
                df_freq = table.df.sort_values("h")
                df_freq["Beta_h"] = beta_from_pf(df_freq.Pf_h.to_numpy())
                if len(table.df) == 1:
                    beta_vals = np.full(water_levels.shape, df_freq.Beta_h.iat[0])
                elif len(table.df) > 1:
                    interpolator = LinearInterpolator(df_freq.h.to_numpy(), df_freq.Beta_h.to_numpy())
                    beta_vals = interpolator.value(water_levels)
                else:
                    continue
                if not np.isinf(beta_vals).all():
                    df_fc_paths[table_name] = beta_vals if plot_beta else pf_from_beta(beta_vals)

            # Also add all failure path fragility curves
            for fc in fcs:
                end_fc = fc.cumulative_probabilities.iloc[:, -1]
                endnode_name = eeg.graph.nodes[end_fc.name]["description"] + f" ({end_fc.name})"
                if plot_beta:
                    df_fc_paths[f"path: {endnode_name}"] = beta_from_pf(end_fc.to_numpy())
                else:
                    df_fc_paths[f"path: {endnode_name}"] = end_fc.to_numpy()
            df_fc_paths[f"scenario: {scen_name}"] = beta_from_pf(pfs) if plot_beta else pfs
            df_fc_paths = pd.DataFrame(df_fc_paths, index=water_levels)

            # Save failure path fragility curves for this scenario
            fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
            df_fc_paths.plot(ax=ax, legend=True)
            if plot_beta:
                ax.set_ylabel("$\\beta$")
            else:
                ax.set_yscale("log")
                ax.set_ylabel("$P_f$")
            ax.set_xlabel("water level [m+NAP]")
            ax.grid()
            fig.savefig(fig_path / f"fc_scenario_{scen_name}.png", bbox_inches="tight")
            plt.close("all")

            # Save scenario results
            for metacol, metaval in eeg.metadata.df.iloc[0].items():
                df_result1[metacol].append(metaval)
            df_result1["Section"].append(section_name)
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

        # Save scenario and combined fragility curve plot
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        df_plot_fc = pd.DataFrame(df_plot_fc).set_index("water level")
        if plot_beta:
            for c in df_plot_fc.columns:
                df_plot_fc[c] = beta_from_pf(df_plot_fc[c].to_numpy())
        df_plot_fc.plot(ax=ax, legend=True)
        if plot_beta:
            ax.set_ylabel("$\\beta$")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("$P_f$")
        ax.set_xlabel("Water level [m+NAP]")
        ax.grid()
        fig.savefig(fig_path / f"fc_section_{section_name}.png", bbox_inches="tight")
        plt.close("all")

    # Write summary result
    df_result1 = pd.DataFrame(df_result1)
    df_result1.to_excel(output_folder / dir_traject.name / f"{dir_traject.name}_result.xlsx")


if __name__ == "__main__":
    main()
