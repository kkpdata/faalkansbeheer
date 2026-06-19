# failure_paths

Python package for loading failure-path event graphs from Excel, storing them in SQLite, and deriving path probability and reliability curves.

## Public entry points

- `ExcelEventGraph`: parse and validate workbook-based scenarios.
- `SqliteEventGraph`: reload stored scenarios from SQLite.
- `EventGraphStore`: persist and query scenario data.

## Install

This package is part of the `failure_paths` Pixi workspace; see the [root README](../../README.md) for setup. There is no supported standalone pip install.

## Example

```python
from failure_paths import ExcelEventGraph

graph = ExcelEventGraph.load("workbook.xlsx", "scenario1")
_, results = graph.get_failure_path_probabilities(water_levels=[3, 4, 5])
```
