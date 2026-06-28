import argparse
import datetime
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sct
import tqdm.auto as tqdm
from failure_paths.common.assemblage import bepaal_N_vak, combine_series
from failure_paths.common.graph_betrouwbaarheidsindex import GraphBetaValuesSingleInteractive
from failure_paths.common.interp import interpolate_beta_curve
from failure_paths.common.prob import INTERPOLATION_BETA_CAP, beta_from_pf, pf_from_beta
from failure_paths.common.traject_normering import TrajectNormering
from failure_paths.reliability import IntegrationConfig, ReliabilityIntegrator
from failure_paths.reliability.curves import FragilityCurve, HazardCurve
from failure_paths.reliability.plotting import plot_failure_histogram, plot_integration_diagnostics_1d
from pandas import DataFrame

from failure_paths import ExcelEventGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and integrate scenario fragility curves.")
    parser.add_argument(
        "--hr-path",
        type=Path,
        default=Path(
            "C:/Users/SAKA/Downloads/DT16-1_hr/WBI2017_Benedenrijn_16-2_v04b"
        ),  # C:\Users\SAKA\Downloads\DT16-1_hr\WBI2017_Benedenrijn_16-2_v04b
    )
    parser.add_argument("--hr-calname", default="ws")
    parser.add_argument(
        "--dir-traject",
        type=Path,
        default=Path(
            "C:/Users/SAKA/Waterschap Rivierenland/Beoordeling Primaire Keringen - LBO2 - 3_Project/Gedeelde informatie/Opleverdossier/99 Assemblage 16-2"
        ),
    )
    parser.add_argument("--output-folder", type=Path, default=Path("C:/Users/SAKA/Downloads/DT16-2_output"))
    parser.add_argument("--scenario-name", default="scenario1")
    parser.add_argument("--wl-min", type=float, default=0.0)
    parser.add_argument("--wl-max", type=float, default=10.0)
    parser.add_argument("--wl-count", type=int, default=101)
    parser.add_argument("--plot-beta", type=bool, default=True)
    parser.add_argument("--plot-tree", type=bool, default=False)
    parser.add_argument("--beta-inf-substitute", type=float, default=None)
    parser.add_argument("--a-vak", type=float, default=0.5, help="Mechanismegevoelige fractie (a) voor bepaling N_vak")
    parser.add_argument(
        "--delta-L", type=float, default=50.0, help="Equivalente onafhankelijke lengte (dL) voor bepaling N_vak"
    )
    parser.add_argument(
        "--dijktraject", type=str, default="16-2"
    )  # hier aanpassen voor dijktraject 16-2 is key naar TRAJECT_PROPERTIES in traject_normering.py
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
    )
    integrator = ReliabilityIntegrator(config=config)
    result = integrator.run(collect_diagnostics_trace=True)

    # visualize 1D integration diagnostics and histogram
    fig, axs = plt.subplots(ncols=2, figsize=(14, 5), dpi=100)
    plot_integration_diagnostics_1d(result, axes=axs[0])

    # Keep the internal water-level resolution, but ensure the outer bins include all failure samples.
    hist_edges = np.asarray(water_levels, dtype=float).copy()
    failure_levels = result.failure_solicitation_levels()
    if failure_levels is not None and failure_levels.size > 0:
        level_min = float(np.min(failure_levels))
        level_max = float(np.max(failure_levels))
        if level_min < hist_edges[0]:
            hist_edges[0] = np.nextafter(level_min, -np.inf)
        if level_max > hist_edges[-1]:
            hist_edges[-1] = np.nextafter(level_max, np.inf)

    # visualize failure probability distribution over water levels
    _, _, hist_data = plot_failure_histogram(
        result, hist_edges, ax=axs[1], conditional=False, solicitation_distribution=integrator.s_distribution
    )
    if scen_name == "combined":
        fig.savefig(fig_path / f"int_section_{scen_name}.png", bbox_inches="tight")
    else:
        fig.savefig(fig_path / f"int_scenario_{scen_name}.png", bbox_inches="tight")
    plt.close("all")

    return result


def _export_graph(df: DataFrame, export_dir: str, dijktraject: str):
    df = df.rename(columns={"M_VAN": "m_start", "M_TOT": "m_end", "dijkvaknummer": "id"})
    df["beta"] = df["Vak_Section_Pf"].apply(lambda x: -1 * sct.norm.ppf(x))
    df_beta_vak = df[["id", "m_start", "m_end", "beta"]]

    beta_traject = -1* sct.norm.ppf(df["Traject_Pf_ondergrens"].iloc[0])

    traject_normering = TrajectNormering(traject_id=dijktraject, norm_is_ondergrens=True)
    GraphBetaValuesSingleInteractive(
        traject_normering=traject_normering, df_beta_vak=df_beta_vak, beta_traject=beta_traject, export_dir=export_dir
    )


def main() -> None:
    args = parse_args()
    hr_path = args.hr_path
    hr_calname = args.hr_calname
    dir_traject = args.dir_traject
    output_folder = args.output_folder
    scenario_name = args.scenario_name
    water_levels = np.linspace(args.wl_min, args.wl_max, args.wl_count)
    plot_beta = args.plot_beta
    plot_tree = args.plot_tree
    beta_inf_sub = args.beta_inf_substitute
    a_vak = args.a_vak
    delta_L = args.delta_L
    dijktraject = args.dijktraject

    # Read all scenarios and structure them into sections
    sections = {}
    meta_cols = []
    for p in dir_traject.rglob("*.xlsx", case_sensitive=False):
        if p.is_file() and not p.name.startswith("~$"):
            try:
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

    print(f"Parsed {len(sections)} sections")
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
            print(f"Processing section '{section_name}', scenario '{scen_name}'")
            # Save tree plot
            if plot_tree:
                eeg.plot(view=False, output_path=fig_path / f"tree_scenario_{scen_name}.png", water_level=6)

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
                df_freq["Beta_h"] = beta_from_pf(df_freq.Pf_h.to_numpy(), inf_substitute=beta_inf_sub)
                if len(table.df) == 1:
                    beta_vals = np.full(water_levels.shape, df_freq.Beta_h.to_numpy()[0])
                elif len(table.df) > 1:
                    beta_cap = float(beta_inf_sub) if beta_inf_sub is not None else float(INTERPOLATION_BETA_CAP)
                    beta_vals = interpolate_beta_curve(
                        df_freq.h.to_numpy(),
                        df_freq.Beta_h.to_numpy(),
                        water_levels,
                        beta_cap=beta_cap,
                    )
                else:
                    continue
                if not np.isinf(beta_vals).all():
                    df_fc_paths[table_name] = beta_vals if plot_beta else pf_from_beta(beta_vals)

            # Also add all failure path fragility curves
            for fc in fcs:
                end_fc = fc.cumulative_probabilities.iloc[:, -1]
                endnode_name = eeg.graph.nodes[end_fc.name]["description"] + f" ({end_fc.name})"
                if plot_beta:
                    df_fc_paths[f"path: {endnode_name}"] = beta_from_pf(end_fc.to_numpy(), inf_substitute=beta_inf_sub)
                else:
                    df_fc_paths[f"path: {endnode_name}"] = end_fc.to_numpy()
                # path_pf = fc.node_probabilities.to_numpy(dtype=float).prod(axis=1)
                # endnode_id = fc.path.nodes[-1]
                # endnode_name = eeg.graph.nodes[endnode_id]["description"] + f" ({endnode_id})"
                # if plot_beta:
                #     df_fc_paths[f"path: {endnode_name}"] = beta_from_pf(path_pf, inf_substitute=beta_inf_sub)
                # else:
                #     df_fc_paths[f"path: {endnode_name}"] = path_pf

            df_fc_paths[f"scenario: {scen_name}"] = beta_from_pf(pfs, inf_substitute=beta_inf_sub) if plot_beta else pfs
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

            # Check that all metadata columns are present.
            missing_metacols = set(meta_cols).difference(eeg.metadata.df.columns.tolist())
            if len(missing_metacols) > 0:
                raise ValueError(
                    f"{section_name=} {scen_name=} misses the following metadata column(s): {missing_metacols}"
                )

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
        if not np.isclose(sum(scen_probs), 1.0):
            raise ValueError(
                f"Scenario probabilities must sum to 1 for section '{section_name}'. Got {sum(scen_probs)}"
            )

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
                df_plot_fc[c] = beta_from_pf(df_plot_fc[c].to_numpy(), inf_substitute=beta_inf_sub)
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
    # Calculate Nvak for each row bases on the length of the vak (LENGTE_VAK), a=1.0, and delta_L (equivalent independent length for STPH)
    df_result1["N_vak"] = np.vectorize(bepaal_N_vak)(df_result1["LENGTE_VAK"], a_vak, delta_L)
    # Upscale to vak level
    df_result1["Vak_Section_Pf"] = df_result1["Section_Pf"] * df_result1["N_vak"]
    # create a list of the first "Vak_Section_Pf" values for each unique Vaknaam
    vak_section_pfs = []
    for vaknaam in df_result1["Vaknaam"].unique():
        vak_section_pfs.append(df_result1[df_result1["Vaknaam"] == vaknaam]["Vak_Section_Pf"].iloc[0])
    # Calculate combined Pf for each Vaknaam using combine_series
    bovengrens_pf, ondergrens_pf = combine_series(vak_section_pfs)
    # add combined Pf to the dataframe (same value for each row)
    df_result1["Traject_Pf_bovengrens"] = bovengrens_pf
    df_result1["Traject_Pf_ondergrens"] = ondergrens_pf
    # calculate percentage of each Vak_Section_Pf relative to the Traject_Pf_bovengrens
    df_result1["Vak_Section_Pf_percentage"] = df_result1["Vak_Section_Pf"] / bovengrens_pf
    #calculate N_traject
    df_result1["N_traject"] = bovengrens_pf / ondergrens_pf
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    df_result1.to_excel(output_folder / dir_traject.name / f"{dir_traject.name}_result_{timestamp}.xlsx", index=False)

    export_dir = output_folder / dir_traject.name
    _export_graph(df=df_result1, export_dir=export_dir, dijktraject=dijktraject)


if __name__ == "__main__":
    main()
