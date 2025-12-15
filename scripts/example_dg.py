from failure_paths import ExcelEventGraph

# Produces error because the excel sheet does not match the indirect mechanism name
eeg = ExcelEventGraph.load("scripts/example_input/voorbeeld_dierlijke_graverij.xlsx", "scenario1")
eeg.plot(view=True, output_path="scripts/example_output/et_dg.png", water_level=3)
