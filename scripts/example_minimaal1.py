from failure_paths import ExcelEventGraph

eeg = ExcelEventGraph.load("scripts/example_input/voorbeeld_minimaal_1_gebeurtenis.xlsx", "scenario1")
eeg.plot(view=True, output_path="scripts/example_output/et_minimaal1.png", water_level=3)
# paths = eeg.get_failure_paths()
_, fcs = eeg.get_failure_path_probabilities(water_levels=[3, 4])

for fc in fcs:
    print(fc.path)
    print(fc.cumulative_probabilities)
