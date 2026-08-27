from failure_paths import ExcelEventGraph

eeg = ExcelEventGraph.load("scripts/example_input/dummy_traject/16-X_006_DP023-DP033.xlsx", "scenario1")
eeg.plot(view=True, output_path="scripts/example_output/et_minimaal.png", water_level=5.6)
# paths = eeg.get_failure_paths()
_, fcs = eeg.get_failure_path_probabilities(water_levels=[1,2,3, 4,5,6,7,8])

for fc in fcs:
    print(fc.path)
    print(fc.cumulative_probabilities)
