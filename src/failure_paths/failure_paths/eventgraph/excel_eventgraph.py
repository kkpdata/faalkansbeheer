from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .eventgraph import EventGraph
from .models import (
    EventTable,
    FrequencyTable,
    GraphEdge,
    GraphNode,
    MetadataTable,
    PathTable,
)

# Excel keyword constants
KEYWORD_METADATA = "metadata"
KEYWORD_PATHS = "faalpadschema"
KEYWORD_EVENTS = "faalpadkansen"


@dataclass
class NormalizedPaths:
    df: pd.DataFrame
    overslag_values: list[str]
    indirect_values: list[str]
    indirect_column: str
    group_columns: list[str]
    group_index_columns: list[str]


@dataclass
class ScenarioGroup:
    start_node: tuple[int, int]
    data: pd.DataFrame


@dataclass
class ScenarioBuildResult:
    nodes: dict[tuple[int, int], GraphNode]
    edges: list[GraphEdge]
    freq_rows: dict[str, list]
    groups: list[ScenarioGroup]


class ExcelEventGraph(EventGraph):
    """Event graph that loads its data from a scenario-oriented Excel workbook."""

    metadata: MetadataTable
    graph: nx.DiGraph
    graph_events: EventTable

    @classmethod
    def load(
        cls,
        excel_path: Path | str,
        scenario_sheet: str,
    ) -> ExcelEventGraph:
        """Load an Excel scenario sheet and build the graph representation.

        Parameters
        ----------
        excel_path : Path | str
            Workbook containing the scenario data.
        scenario_sheet : str
            Name of the sheet holding the metadata/paths/events tables.

        Returns
        -------
        ExcelEventGraph
            Populated graph instance backed by the supplied workbook.
        """
        excel_path = Path(excel_path)
        metadata, paths, events, freq_tables = ExcelEventGraph._find_tables(
            excel_path,
            scenario_sheet,
        )
        digraph, graph_events = ExcelEventGraph._build_digraph(
            paths,
            events,
            freq_tables,
            f"{excel_path.stem}: {scenario_sheet}",
        )

        return cls(
            metadata=metadata,
            graph=digraph,
            graph_events=graph_events,
        )

    @staticmethod
    def _build_digraph(
        paths: PathTable,
        events: EventTable,
        freq_tables: dict[str, FrequencyTable],
        start_node: str,
    ) -> nx.DiGraph:
        """Translate normalized tables into a directed graph.

        Parameters
        ----------
        paths : PathTable
            Failure path definitions.
        events : EventTable
            Table containing Pf/Beta values.
        freq_tables : dict[str, FrequencyTable]
            Optional exceedance-frequency tables keyed by scenario attribute.
        start_node : str
            Label used for the root node.

        Returns
        -------
        nx.DiGraph
            Directed event graph with node metadata and EventTable.
        """
        normalized = ExcelEventGraph._normalize_paths_table(paths)
        scenario_build = ExcelEventGraph._build_scenario_nodes(
            normalized=normalized,
            freq_tables=freq_tables,
            start_node=start_node,
            required_columns=events.required_columns,
        )
        ExcelEventGraph._build_event_nodes(scenario_build)

        df_events = events.df.reset_index()[list(events.required_columns)]
        df_freq_events = pd.DataFrame(scenario_build.freq_rows).astype(df_events.dtypes)
        if not df_freq_events.empty:
            df_freq_events = df_freq_events.drop_duplicates(subset=["Faalpad_ID", "Knoop_ID", "h"])
        df_full_events = pd.concat([df_events, df_freq_events], ignore_index=True)
        graph_events = EventTable.from_dataframe(df_full_events)

        edges = list(set(scenario_build.edges))
        event_nids = graph_events.get_event_ids()
        graph = nx.DiGraph()
        for node_id, node in scenario_build.nodes.items():
            if node.node_type != "start_node" and node_id not in event_nids:
                raise ValueError(
                    f"[{start_node}] {node_id=} '{node.description}' has no failure probability; "
                    "ensure the event table contains matching Pf/Beta rows."
                )
            graph.add_node(
                node.node_id,
                description=node.description,
                node_type=node.node_type,
                type_name=node.type_name,
            )
        graph.add_edges_from((edge.source, edge.target) for edge in edges)

        return graph, graph_events

    @staticmethod
    def _normalize_paths_table(paths: PathTable) -> NormalizedPaths:
        """Normalize the raw path table and derive grouping metadata.

        Parameters
        ----------
        paths : PathTable
            Input table describing scenario combinations and events.

        Returns
        -------
        NormalizedPaths
            Container with the normalized dataframe and group metadata.
        """
        df_paths = paths.df.copy()
        uniq_overslag = df_paths.Overslag.unique().tolist()
        uniq_indirect = df_paths.Indirect_mechanisme.unique().tolist()
        indirect_type = "geen indirect mechanisme"
        if len(uniq_indirect) == 2:
            indirect_type = [idt for idt in uniq_indirect if idt != "nee"][0]

        df_paths = df_paths.rename(columns={"Indirect_mechanisme": indirect_type})
        df_paths.loc[df_paths[indirect_type] != "nee", indirect_type] = "ja"
        uniq_indirect = df_paths[indirect_type].unique().tolist()

        colmapping = {"Initiele_gebeurtenis": 1}
        for col in df_paths.columns:
            if col.startswith("Vervolggebeurtenis_"):
                colmapping[col] = int(col.replace("Vervolggebeurtenis_", "")) + 1
        df_paths = df_paths.rename(columns=colmapping)

        df_paths = df_paths.set_index("Faalpad_ID")
        group_columns = ["Overslag", indirect_type]
        group_index_columns = [f"{g}_idx" for g in group_columns]
        for group_idx, group_col in zip(group_index_columns, group_columns):
            df_paths[group_idx], _ = pd.factorize(df_paths[group_col])

        return NormalizedPaths(
            df=df_paths,
            overslag_values=uniq_overslag,
            indirect_values=uniq_indirect,
            indirect_column=indirect_type,
            group_columns=group_columns,
            group_index_columns=group_index_columns,
        )

    @staticmethod
    def _build_scenario_nodes(
        normalized: NormalizedPaths,
        freq_tables: dict[str, FrequencyTable],
        start_node: str,
        required_columns: tuple[str, ...],
    ) -> ScenarioBuildResult:
        """Create scenario nodes and edges for every overslag/indirect group.

        Parameters
        ----------
        normalized : NormalizedPaths
            Preprocessed path information returned by :meth:`_normalize_paths_table`.
        freq_tables : dict[str, FrequencyTable]
            Frequency tables used to populate Pf rows for each scenario dimension.
        start_node : str
            Label applied to the root node.
        required_columns : tuple[str, ...]
            Required column order used when populating frequency rows.

        Returns
        -------
        ScenarioBuildResult
            Nodes, edges, frequency samples, and grouped event data.
        """
        root_id = (-3, 0)
        nodes: dict[tuple[int, int], GraphNode] = {
            root_id: GraphNode(node_id=root_id, description=start_node, node_type="start_node", type_name=None)
        }
        edges: list[GraphEdge] = []
        freq_rows = {col: [] for col in required_columns}
        groups: list[ScenarioGroup] = []
        group_keys = normalized.group_columns + normalized.group_index_columns

        for (overslag, indirect, overslag_idx, indirect_idx), group in normalized.df.groupby(group_keys):
            overslag_idx = int(overslag_idx)
            indirect_idx = int(indirect_idx)
            cnode1 = (-2, overslag_idx)
            cnode2 = (-1, indirect_idx)
            nodes[cnode1] = GraphNode(
                node_id=cnode1,
                description=f"overslag: {overslag}",
                node_type="scenario_node",
                type_name="overslag",
            )
            nodes[cnode2] = GraphNode(
                node_id=cnode2,
                description=f"{normalized.indirect_column}: {indirect}",
                node_type="scenario_node",
                type_name=normalized.indirect_column,
            )
            edges.append(GraphEdge(source=root_id, target=cnode1))
            edges.append(GraphEdge(source=cnode1, target=cnode2))

            for uniq_vals, cval, ctype, fid, kid in zip(
                [normalized.overslag_values, normalized.indirect_values],
                [overslag, indirect],
                ["overslag", normalized.indirect_column],
                [-2, -1],
                [overslag_idx, indirect_idx],
            ):
                if len(uniq_vals) == 1:
                    freq_rows["Faalpad_ID"].append(fid)
                    freq_rows["Knoop_ID"].append(kid)
                    freq_rows["h"].append(0)
                    freq_rows["Pf_h"].append(1.0)
                    freq_rows["Beta_h"].append(np.nan)
                else:
                    table = freq_tables.get(ctype)
                    if table is None:
                        raise ValueError(
                            f"Frequency table for '{ctype}' not found while processing scenario '{start_node}'"
                        )
                    for row in table.df.itertuples():
                        freq_rows["Faalpad_ID"].append(fid)
                        freq_rows["Knoop_ID"].append(kid)
                        freq_rows["h"].append(row.h)
                        pf_val = row.Pf_h if cval == "ja" else 1.0 - row.Pf_h
                        freq_rows["Pf_h"].append(pf_val)
                        freq_rows["Beta_h"].append(np.nan)

            event_group = group.drop(columns=group_keys)
            groups.append(ScenarioGroup(start_node=cnode2, data=event_group))

        return ScenarioBuildResult(
            nodes=nodes,
            edges=edges,
            freq_rows=freq_rows,
            groups=groups,
        )

    @staticmethod
    def _build_event_nodes(result: ScenarioBuildResult) -> None:
        """Expand each scenario group into event nodes and edges.

        Parameters
        ----------
        result : ScenarioBuildResult
            Mutable container populated by :meth:`_build_scenario_nodes`.
        """
        for scenario_group in result.groups:
            for faalpad_id, row in scenario_group.data.iterrows():
                prev_node = scenario_group.start_node
                row_nonna = row.dropna(how="any")
                for i, (nid, desc) in enumerate(row_nonna.items(), start=1):
                    enode = (int(faalpad_id), int(nid))
                    result.nodes[enode] = GraphNode(
                        node_id=enode,
                        description=desc,
                        node_type="failure_node" if i == len(row_nonna) else "event_node",
                        type_name=None,
                    )
                    result.edges.append(GraphEdge(source=prev_node, target=enode))
                    prev_node = enode

    @staticmethod
    def _find_tables(
        excel_path: Path,
        scenario_sheet: str,
    ) -> tuple[MetadataTable, PathTable, EventTable, dict[str, FrequencyTable] | None]:
        """Locate the metadata/path/event tables within the workbook.

        Parameters
        ----------
        excel_path : Path
            Workbook path.
        scenario_sheet : str
            Sheet name containing the required tables.

        Returns
        -------
        tuple
            Metadata, path, event tables, plus frequency tables if present.

        Raises
        ------
        ValueError
            If required sections are missing or duplicated.
        """
        with pd.ExcelFile(excel_path, engine="openpyxl") as xlsx:
            offset_mapping = ExcelEventGraph._find_table_offsets(xlsx, sheet_name=scenario_sheet)

            table_mapping: dict[str, Any] = {
                KEYWORD_METADATA: MetadataTable,
                KEYWORD_PATHS: PathTable,
                KEYWORD_EVENTS: EventTable,
            }
            for i, (key, idx) in enumerate(offset_mapping.items()):
                idx_start = idx + 1
                nrows = None
                if i < len(offset_mapping) - 1:
                    idx_end = offset_mapping[list(offset_mapping.keys())[i + 1]] - 1
                    nrows = idx_end - idx_start

                df_part = pd.read_excel(xlsx, sheet_name=scenario_sheet, skiprows=idx_start, nrows=nrows)
                df_part = df_part.loc[:, ~df_part.columns.str.startswith("Unnamed:")]
                table_mapping[key] = table_mapping[key].from_dataframe(df_part)

            freq_tables = ExcelEventGraph._load_frequency_tables(xlsx)

        return (
            table_mapping[KEYWORD_METADATA],
            table_mapping[KEYWORD_PATHS],
            table_mapping[KEYWORD_EVENTS],
            freq_tables,
        )

    @staticmethod
    def _load_frequency_tables(xlsx: pd.ExcelFile) -> dict[str, FrequencyTable]:
        """Load frequency tables from sheets prefixed with ``FP_``.

        Parameters
        ----------
        xlsx : pd.ExcelFile
            Open workbook handle reused while parsing other tables.

        Returns
        -------
        dict[str, FrequencyTable]
            Mapping of frequency table type to validated :class:`FrequencyTable`.
        """
        freq_tables: dict[str, FrequencyTable] = {}
        for sheet_name in xlsx.sheet_names:
            if not sheet_name.upper().startswith("FP_"):
                continue
            freq_type = sheet_name[3:].strip().lower()
            if not freq_type:
                continue
            df_freq = pd.read_excel(xlsx, sheet_name=sheet_name)
            freq_tables[freq_type] = FrequencyTable.from_dataframe(df_freq)
        return freq_tables

    @staticmethod
    def _find_table_offsets(
        xlsx: pd.ExcelFile,
        sheet_name: str,
    ) -> dict[str, int]:
        """Return row offsets where the metadata/path/event tables start."""
        df_offsets = pd.read_excel(xlsx, sheet_name=sheet_name, usecols="A", header=None)
        df_offsets = df_offsets.iloc[:, 0].str.lower()

        offset_mapping = {
            KEYWORD_METADATA: None,
            KEYWORD_PATHS: None,
            KEYWORD_EVENTS: None,
        }

        issues: list[str] = []
        for keyword in offset_mapping.keys():
            loc_key = df_offsets == keyword
            count = int(loc_key.sum())
            if count == 0:
                issues.append(f"missing keyword '{keyword}'")
                continue
            if count > 1:
                issues.append(f"keyword '{keyword}' appears {count} times")
                continue
            offset_mapping[keyword] = int(np.nonzero(loc_key)[0][0])

        if issues:
            joined = "; ".join(issues)
            raise ValueError(f"Cannot determine table offsets for sheet '{sheet_name}': {joined}")

        return offset_mapping
