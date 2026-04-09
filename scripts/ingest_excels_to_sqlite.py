from __future__ import annotations

from pathlib import Path

import numpy as np
from failure_paths.eventgraph import EventGraphStore, ExcelEventGraph, SqliteEventGraph

# ---------------------------------------------------------------------------
# Configuration – adjust these values before running the script.
DB_PATH = Path("scripts/example_output/failure_paths.db")
SECTION = "west"
SHEETS = ["scenario1"]
EXCELS = [
    Path("scripts/example_input/AW172_voorbeeld_GD.xlsx"),
]
TAGS: str | None = None
WATER_LEVELS = np.linspace(0, 10, 11)  # Example values, adjust as needed
# ---------------------------------------------------------------------------


def ingest_scenario(
    store: EventGraphStore,
    excel_path: Path,
    sheet_name: str,
    *,
    section: str,
    tags: str | None,
) -> None:
    """Parse a workbook/sheet pair and persist the scenario."""
    metadata, paths, events, freq_tables = ExcelEventGraph._find_tables(excel_path, sheet_name)
    start_label = f"{excel_path.stem}: {sheet_name}"
    digraph, graph_events = ExcelEventGraph._build_digraph(
        paths,
        events,
        freq_tables,
        start_label,
    )

    with store.transaction() as conn:
        scenario_id = store.upsert_scenario(
            section=section,
            source_path=str(excel_path),
            sheet_name=sheet_name,
            start_node_label=start_label,
            tags=tags,
            conn=conn,
        )

        store.write_dataframe("metadata", metadata.to_required_records(), scenario_id, conn=conn)
        store.write_dataframe("events", graph_events.to_required_records(), scenario_id, conn=conn)
        store.write_graph(digraph, scenario_id, conn=conn)
        store.write_frequency_tables(freq_tables, scenario_id, conn=conn)


if __name__ == "__main__":
    errors: list[str] = []

    with EventGraphStore(DB_PATH) as store:
        for excel in EXCELS:
            excel_path = Path(excel).resolve()
            if not excel_path.exists():
                errors.append(f"{excel_path} does not exist")
                continue

            for sheet_name in SHEETS:
                try:
                    ingest_scenario(store, excel_path, sheet_name, section=SECTION, tags=TAGS)
                    print(f"[OK] {excel_path.name} / {sheet_name}")
                except Exception as exc:
                    errors.append(f"{excel_path.name}/{sheet_name}: {exc}")
                    print(f"[ERR] {excel_path.name} / {sheet_name}: {exc}")

        # -------------------------------------------------------------------
        # Example: read back the graph using SqliteEventGraph.
        try:
            sample = store.list_scenarios()[0]
        except IndexError:
            pass
        else:
            restored = SqliteEventGraph.load(DB_PATH, scenario_id=sample.id)
            restored.plot(view=True, output_path="scripts/example_output/event_tree.png", water_level=3)
            print(
                "\nExample read-back -> "
                f"scenario_id={restored.scenario_id} ({sample.start_node_label}) "
                f"nodes={restored.graph.number_of_nodes()} "
                f"edges={restored.graph.number_of_edges()}"
            )
            paths = restored.get_failure_paths()
            for p in paths:
                print(p)
            _, results = restored.get_failure_path_probabilities(water_levels=WATER_LEVELS)
            for r in results:
                print(r.cumulative_probabilities)

    if errors:
        print("\nIngestion completed with errors:")
        for err in errors:
            print(f" - {err}")
