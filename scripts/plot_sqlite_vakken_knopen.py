from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from failure_paths.common.prob import beta_from_pf

from failure_paths import EventGraph, EventGraphStore, SqliteEventGraph


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SQLite section/node plotting.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Load eventgraph data from SQLite, select sections and nodes, "
            "build merged path-probability dataframes, and save a plot to file."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the SQLite database.",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Comma-separated list of sections (scenarios.section). Omit to include all sections.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Comma-separated list of node names (graph_nodes.description). Omit to include all nodes.",
    )
    parser.add_argument(
        "--metric",
        choices=["beta", "pf"],
        default="beta",
        help="Curve metric to plot. Defaults to 'beta'.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output PNG file path.",
    )
    return parser.parse_args()


def _parse_csv_values(raw_values: str | None) -> list[str]:
    """Parse comma-separated values into a unique ordered list.

    Parameters
    ----------
    raw_values : str | None
        Comma-separated values from CLI input.

    Returns
    -------
    list[str]
        Unique stripped values in original order.
    """
    if raw_values is None:
        return []
    parsed = [item.strip() for item in raw_values.split(",") if item.strip()]
    return list(dict.fromkeys(parsed))


def _format_available(values: Sequence[str], *, limit: int = 30) -> str:
    """Format available options for clear error messages.

    Parameters
    ----------
    values : Sequence[str]
        Candidate values to display.
    limit : int, optional
        Maximum number of displayed values before truncation.

    Returns
    -------
    str
        Comma-separated values plus truncation indicator when needed.
    """
    if not values:
        return "<none>"
    if len(values) <= limit:
        return ", ".join(values)
    head = ", ".join(values[:limit])
    return f"{head}, ... ({len(values) - limit} more)"


def _validate_selection(
    store: EventGraphStore,
    *,
    sections: Sequence[str],
    node_names: Sequence[str],
) -> None:
    """Validate section and node-name selections against the database.

    Parameters
    ----------
    store : EventGraphStore
        Open SQLite store used for validation.
    sections : Sequence[str]
        Requested section names.
    node_names : Sequence[str]
        Requested node-name selections.

    Raises
    ------
    ValueError
        If one or more sections or node names are not available.
    """
    available_sections = store.list_sections()
    available_sections_set = set(available_sections)
    invalid_sections = [section for section in sections if section not in available_sections_set]
    if invalid_sections:
        raise ValueError(
            f"Unknown sections: {invalid_sections}. Available sections: {_format_available(available_sections)}"
        )

    available_nodes = store.list_node_names(sections=sections)
    available_nodes_set = set(available_nodes)
    invalid_nodes = [node_name for node_name in node_names if node_name not in available_nodes_set]
    if invalid_nodes:
        raise ValueError(
            "Unknown nodes: "
            f"{invalid_nodes}. Available nodes for selected sections: "
            f"{_format_available(available_nodes)}"
        )


def _build_column_order(
    df_weighted: pd.DataFrame,
    *,
    sections: Sequence[str],
    node_names: Sequence[str],
) -> list[str]:
    """Build deterministic output column ordering.

    Parameters
    ----------
    df_weighted : pd.DataFrame
        Aggregated dataframe with one row per section/node/h.
    sections : Sequence[str]
        Requested section order from CLI.
    node_names : Sequence[str]
        Requested node-name order from CLI.

    Returns
    -------
    list[str]
        Ordered output column labels.
    """
    section_order = {section: idx for idx, section in enumerate(sections)}
    node_order = {node_name: idx for idx, node_name in enumerate(node_names)}

    order_df = df_weighted.loc[:, ["column_label", "section", "node_name", "faalpad_id", "knoop_id"]].drop_duplicates()
    order_df["section_order"] = order_df["section"].map(section_order).fillna(len(section_order))
    order_df["node_order"] = order_df["node_name"].map(node_order).fillna(len(node_order))

    order_df = order_df.sort_values(
        ["section_order", "node_order", "faalpad_id", "knoop_id", "column_label"],
        kind="stable",
    )
    return order_df["column_label"].tolist()


def build_plot_dataframes(
    store: EventGraphStore,
    *,
    sections: Sequence[str],
    node_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build wide Pf_h and Beta_h dataframes from combined eventgraph paths.

    Parameters
    ----------
    store : EventGraphStore
        Open SQLite store in read-only or read-write mode.
    sections : Sequence[str]
        Selected section names (``scenarios.section``).
    node_names : Sequence[str]
        Selected node descriptions (``graph_nodes.description``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Two wide dataframes ``(df_pf_wide, df_beta_wide)`` with:
        - index: ``h``
        - columns: ``"{section} | {node} ({faalpad},{knoop_id})"``
        Values are path-based probabilities and derived reliability indices.

    Raises
    ------
    ValueError
        If available selections are empty, selections are invalid, water levels
        are missing, path data is unavailable, or grouped scenario weights are
        invalid.
    RuntimeError
        If a scenario cannot be loaded from SQLite.
    """
    resolved_sections = list(dict.fromkeys(sections))
    if not resolved_sections:
        resolved_sections = store.list_sections()
    if not resolved_sections:
        raise ValueError("No sections are available in the SQLite database.")

    resolved_nodes = list(dict.fromkeys(node_names))
    if not resolved_nodes:
        resolved_nodes = store.list_node_names(sections=resolved_sections)
    if not resolved_nodes:
        raise ValueError("No node names are available for the selected sections.")

    # 1) Resolve and validate requested filters against available SQLite content.
    _validate_selection(store, sections=resolved_sections, node_names=resolved_nodes)
    df_selection = store.fetch_selected_scenario_nodes(
        sections=resolved_sections,
        node_names=resolved_nodes,
    )
    if df_selection.empty:
        raise ValueError(
            "No scenario/node rows found for selection: "
            f"sections={list(resolved_sections)}, nodes={list(resolved_nodes)}."
        )
    df_selection["scenario_weight"] = pd.to_numeric(df_selection["scenario_weight"], errors="coerce")
    if df_selection["scenario_weight"].isna().any():
        raise ValueError("ScenarioKans contains missing or non-numeric values.")

    df_selection["scenario_id"] = pd.to_numeric(df_selection["scenario_id"], errors="coerce").astype(int)
    df_selection["faalpad_id"] = pd.to_numeric(df_selection["faalpad_id"], errors="coerce").astype(int)
    df_selection["knoop_id"] = pd.to_numeric(df_selection["knoop_id"], errors="coerce").astype(int)

    scenario_df = df_selection.loc[:, ["scenario_id", "section", "scenario_weight"]].drop_duplicates()
    scenario_ids = scenario_df["scenario_id"].tolist()
    water_levels = store.fetch_water_levels_for_scenarios(scenario_ids)
    if water_levels.size == 0:
        raise ValueError("No water levels found in events for selected scenarios.")

    # 2) Build per-scenario, per-node path-based Pf curves on a shared water-level grid.
    path_curve_frames: list[pd.DataFrame] = []
    db_path = store.db_path
    for scenario_row in scenario_df.itertuples(index=False):
        scenario_id = int(scenario_row.scenario_id)
        section = str(scenario_row.section)
        scenario_weight = float(scenario_row.scenario_weight)
        try:
            sqlite_graph = SqliteEventGraph.load(db_path, scenario_id=scenario_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to load scenario_id={scenario_id} from SQLite path '{db_path}'.") from exc

        _, path_results = sqlite_graph.get_failure_path_probabilities(water_levels=water_levels)
        scenario_nodes = (
            df_selection.loc[
                df_selection["scenario_id"] == scenario_id,
                ["node_name", "faalpad_id", "knoop_id"],
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        for node_row in scenario_nodes.itertuples(index=False):
            node_name = str(node_row.node_name)
            faalpad_id = int(node_row.faalpad_id)
            knoop_id = int(node_row.knoop_id)
            try:
                node_curve = sqlite_graph.aggregate_combined_path_probabilities(
                    path_results,
                    include_nodes=(faalpad_id, knoop_id),
                    clip=True,
                )
            except ValueError as exc:
                raise ValueError(
                    "Failed to construct a combined-path curve for "
                    f"scenario_id={scenario_id}, node_id=({faalpad_id},{knoop_id})."
                ) from exc
            path_curve_frames.append(
                pd.DataFrame(
                    {
                        "scenario_id": scenario_id,
                        "section": section,
                        "node_name": node_name,
                        "faalpad_id": faalpad_id,
                        "knoop_id": knoop_id,
                        "scenario_weight": scenario_weight,
                        "h": water_levels,
                        "path_pf_h": node_curve,
                    }
                )
            )

    if not path_curve_frames:
        raise ValueError("No path-based node curves could be constructed for the selection.")

    # 3) Enforce strict scenario-weight consistency before cross-scenario merging.
    df_paths = pd.concat(path_curve_frames, ignore_index=True)
    group_cols = ["section", "node_name", "faalpad_id", "knoop_id", "h"]
    required_columns = set(group_cols) | {"path_pf_h", "scenario_id", "section", "scenario_weight"}
    missing_columns = sorted(required_columns.difference(df_paths.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for weighted merge: {missing_columns}")

    scenario_weights = df_paths.loc[:, ["section", "scenario_id", "scenario_weight"]].copy()
    if scenario_weights.empty:
        raise ValueError("No scenario weights are available for weighted merge.")
    scenario_weights["scenario_weight"] = pd.to_numeric(scenario_weights["scenario_weight"], errors="coerce")
    if scenario_weights["scenario_weight"].isna().any():
        raise ValueError("ScenarioKans contains missing or non-numeric values.")

    unique_weight_counts = scenario_weights.groupby(
        ["section", "scenario_id"],
        sort=False,
    )["scenario_weight"].nunique(dropna=False)
    if (unique_weight_counts > 1).any():
        offenders = unique_weight_counts[unique_weight_counts > 1].index.tolist()
        details = ", ".join(f"({section}, {scenario_id})" for section, scenario_id in offenders)
        raise ValueError(f"ScenarioKans must be unique per (section, scenario_id). Conflicts: {details}")

    scenario_weights = scenario_weights.groupby(
        ["section", "scenario_id"],
        as_index=False,
        sort=False,
    )["scenario_weight"].first()
    section_sums = scenario_weights.groupby("section", sort=False)["scenario_weight"].sum()
    invalid_mask = ~np.isclose(section_sums.to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-9)
    if invalid_mask.any():
        invalid_sections = section_sums.index[invalid_mask]
        details = ", ".join(f"{section}={section_sums.loc[section]:.12g}" for section in invalid_sections)
        raise ValueError(f"Scenario probabilities must sum to 1 per section. Got: {details}")

    values = pd.to_numeric(df_paths["path_pf_h"], errors="coerce")
    if values.isna().any():
        raise ValueError("Column 'path_pf_h' contains missing or non-numeric values.")

    weighted_df = df_paths.copy()
    weighted_df["path_pf_h"] = values
    weighted_df["scenario_weight"] = pd.to_numeric(weighted_df["scenario_weight"], errors="raise")

    def _aggregate_group(group: pd.DataFrame) -> float:
        merged = EventGraph.aggregate_weighted_scenario_curves(
            curves=[np.array([value], dtype=float) for value in group["path_pf_h"].to_numpy(dtype=float)],
            weights=group["scenario_weight"].to_numpy(dtype=float),
            require_sum_one=False,
        )
        return float(merged[0])

    # 4) Produce the final section/node curves and reshape to plotting-friendly wide frames.
    df_weighted = weighted_df.groupby(group_cols, sort=False).apply(_aggregate_group).rename("pf_h").reset_index()

    df_weighted["column_label"] = df_weighted.apply(
        lambda row: f"{row['section']} | {row['node_name']} ({int(row['faalpad_id'])},{int(row['knoop_id'])})",
        axis=1,
    )

    ordered_columns = _build_column_order(df_weighted, sections=resolved_sections, node_names=resolved_nodes)
    df_pf_wide = df_weighted.pivot_table(
        index="h",
        columns="column_label",
        values="pf_h",
        aggfunc="first",
    )
    df_pf_wide = df_pf_wide.sort_index().reindex(columns=ordered_columns)
    df_pf_wide.index.name = "h"

    beta_values = beta_from_pf(df_pf_wide.to_numpy(dtype=float), tail="upper")
    df_beta_wide = pd.DataFrame(beta_values, index=df_pf_wide.index, columns=df_pf_wide.columns)
    df_beta_wide.index.name = "h"
    return df_pf_wide, df_beta_wide


def main() -> None:
    """Run the CLI entrypoint for SQLite section/node plotting."""
    args = parse_args()
    sections = _parse_csv_values(args.sections)
    node_names = _parse_csv_values(args.nodes)

    # End goal A: load and merge selected SQLite scenarios into aligned Pf/Beta tables.
    with EventGraphStore(args.db_path, read_only=True) as store:
        df_pf_wide, df_beta_wide = build_plot_dataframes(
            store,
            sections=sections,
            node_names=node_names,
        )

    # End goal B: render exactly one requested metric and persist it as a figure.
    use_beta = args.metric == "beta"
    df_plot = df_beta_wide if use_beta else df_pf_wide
    ylabel = "Beta_h [-]" if use_beta else "Pf_h [-]"
    title = (
        "Derived Beta_h for selected section/node combinations"
        if use_beta
        else "Merged path-based Pf_h for selected section/node combinations"
    )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 6), constrained_layout=True)
    for column in df_plot.columns:
        ax.plot(df_plot.index, df_plot[column], label=column)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("h [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    output_path = args.output_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
