from failure_paths import ExcelEventGraph

eeg = ExcelEventGraph.load("scripts/example_input/AW172_voorbeeld_GD.xlsx", "scenario1")
eeg.plot(view=True, output_path="scripts/example_output/event_tree.png", water_level=3)
paths = eeg.get_failure_paths()
eeg
