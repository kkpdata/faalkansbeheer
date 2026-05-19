import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and integrate scenario fragility curves.")
    parser.add_argument("--hr-path", type=Path, default=Path("scripts/example_input/dummy_hr"))
    parser.add_argument("--hr-calname", default="ws")
    parser.add_argument("--dir-traject", type=Path, default=Path("scripts/example_input/dummy_traject"))
    parser.add_argument("--output-folder", type=Path, default=Path("scripts/example_output"))
    parser.add_argument("--scenario-name", default="scenario1")
    parser.add_argument("--wl-min", type=float, default=0.0)
    parser.add_argument("--wl-max", type=float, default=10.0)
    parser.add_argument("--wl-count", type=int, default=101)
    parser.add_argument("--plot-beta", type=bool, default=True)
    parser.add_argument("--plot-tree", type=bool, default=True)
    parser.add_argument("--beta-inf-substitute", type=float, default=None)
    parser.add_argument("--a-vak", type=float, default=0.5, help="Mechanismegevoelige fractie (a) voor bepaling N_vak")
    parser.add_argument(
        "--delta-L", type=float, default=50.0, help="Equivalente onafhankelijke lengte (dL) voor bepaling N_vak"
    )
    parser.add_argument("--dijktraject", type=str, default="16-1")
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help="Optional SQLite database path to persist ingested scenario eventgraphs during assemble.",
    )
    return parser.parse_args()
