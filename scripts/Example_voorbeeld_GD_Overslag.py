from pathlib import Path
from failure_paths import ExcelEventGraph
import numpy as np

range_water_levels = [6,7,8,9,10,11,12,13,14,15]

# generate file list for all excel files in the example_input directory
input_folder = Path("scripts/example_input")
# file_list = list(input_folder.glob("*.xlsx"))
file_list = ["voorbeeld_GD_Overslag.xlsx"]


eeg = ExcelEventGraph.load("scripts/example_input/voorbeeld_GD_Overslag.xlsx", "scenario1")

# iterate over all water levels and files in the directory example_input
for file in file_list:
    fcs = eeg.get_failure_path_probabilities(water_levels=range_water_levels)
    for water_level in range_water_levels:
        eeg.plot(view=False, output_path=f"scripts/example_output/voorbeeld_DG_Overslag_waterlevel_{water_level}.png", water_level=water_level)
        
for fc in fcs:
    print(fc.path)
    print(fc.cumulative_probabilities)