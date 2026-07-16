import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a scatter plot of beta points from an Excel file.")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path(r"C:\Users\SAKA\Downloads\DT16-1_output\99 Assemblage 16-1\verzameling_prob_sommen_dt16-1.xlsx"),
        help="Path to the input Excel file containing the data.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path(r"C:\Users\SAKA\Downloads\DT16-1_output\99 Assemblage 16-1\scatter_plot.png"),
        help="Path to save the output scatter plot image.",
    )
    parser.add_argument("--x_column", type=str, default="% kerende hoogte", help="Column name for X-axis values.")
    parser.add_argument("--y_column", type=str, default="beta", help="Column name for Y-axis values.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load the data from excel file
    data = pd.read_excel(args.input_file, skiprows=2)  # Skip the first row if it contains headers or metadata

    required_columns = {args.x_column, args.y_column, "Vak", "situatie"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in input file: {sorted(missing_columns)}")

    plot_data = data.copy()
    plot_data["Vak_label"] = plot_data["Vak"].fillna("Onbekend").astype(str)
    plot_data["Situatie_label"] = plot_data["situatie"].fillna("Onbekend").astype(str)

    vak_values = sorted(plot_data["Vak_label"].unique())
    situatie_values = sorted(plot_data["Situatie_label"].unique())

    cmap = plt.get_cmap("tab20", max(len(vak_values), 1))
    vak_to_color = {vak: cmap(i) for i, vak in enumerate(vak_values)}

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    situatie_to_marker = {
        situatie: markers[i % len(markers)] for i, situatie in enumerate(situatie_values)
    }
    situatie_marker_size = 70

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a scatter plot
    for situatie, situatie_group in plot_data.groupby("Situatie_label"):
        marker = situatie_to_marker[situatie]
        for vak, vak_group in situatie_group.groupby("Vak_label"):
            color = vak_to_color[vak]
            ax.scatter(
                vak_group[args.x_column],
                vak_group[args.y_column],
                color=color,
                marker=marker,
                s=situatie_marker_size,
                alpha=0.9,
            )
    # set basislijn
    ax.plot([0.0,0.25,0.50,0.75,1.0],[10,6,4.0,1.5,1.0], color='black', linestyle='--', linewidth=1, marker='o', markersize=6, label='Geometrische beoordeling')
    # aangescherpte fragilitycurve
    ax.plot([0.0,0.25,0.50,0.75,1.0],[10,6,4.0,2.5,2.0], color='red', linestyle='--', linewidth=1, marker='o', markersize=6, label='Aangescherpte fragilitycurve')


    ax.set_xlabel(args.x_column)
    ax.set_ylabel(args.y_column)
    ax.set_ylim(bottom=0, top=20)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(2))
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.grid(True)

    vak_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=vak_to_color[vak],
            markersize=8,
            label=vak,
        )
        for vak in vak_values
    ]

    situatie_handles = [
        mlines.Line2D(
            [0],
            [0],
            marker=situatie_to_marker[situatie],
            color="black",
            linestyle="None",
            markersize=10,
            label=situatie,
        )
        for situatie in situatie_values
    ]

    lijn_handles = [
        mlines.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1,
            marker="o",
            markersize=6,
            label="Uitgangspunt geometrische methode",
        ),
        mlines.Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=1,
            marker="o",
            markersize=6,
            label="Aangescherpte geometrische methode",
        ),
    ]

    lijn_legend = ax.legend(
        handles=lijn_handles,
        loc="upper right",
    )
    ax.add_artist(lijn_legend)

    vak_legend = ax.legend(
        handles=vak_handles,
        title="Vak (kleur)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=max(1, min(4, len(vak_handles))),
    )
    ax.add_artist(vak_legend)

    ax.legend(
        handles=situatie_handles,
        title="Situatie (vorm)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=max(1, min(4, len(situatie_handles))),
    )
    plt.subplots_adjust(bottom=0.35)

    # Show the plot
    plt.show()
