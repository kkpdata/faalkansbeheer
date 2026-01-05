from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from failure_paths import ExcelEventGraph

range_water_levels = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# generate file list for all excel files in the example_input directory
input_folder = Path("scripts/example_input")
# file_list = list(input_folder.glob("*.xlsx"))
file_list = ["voorbeeld_GD_Overslag.xlsx"]


eeg = ExcelEventGraph.load("scripts/example_input/voorbeeld_GD_Overslag.xlsx", "scenario1")

# iterate over all water levels and files in the directory example_input
for file in file_list:
    _, fcs = eeg.get_failure_path_probabilities(water_levels=range_water_levels)
    for water_level in range_water_levels:
        eeg.plot(
            view=False,
            output_path=f"scripts/example_output/voorbeeld_DG_Overslag_waterlevel_{water_level}.png",
            water_level=water_level,
        )

    df_all = {}
    for fc in fcs:
        end_fc = fc.cumulative_probabilities.iloc[:, -1]
        endnode_name = eeg.graph.nodes[end_fc.name]["description"] + f" ({end_fc.name})"
        # df_all[endnode_name] = scipy.stats.norm.isf(end_fc.to_numpy())
        df_all[endnode_name] = end_fc.to_numpy()
    df_all = pd.DataFrame(df_all, index=range_water_levels)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    df_all.plot(ax=ax, logy=True)
    fig.tight_layout()
    plt.show()
