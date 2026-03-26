from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import ClassVar, Self

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class TableLoadContext:
    """Origin metadata used to enrich dataframe validation errors."""

    workbook_path: str | Path | None = None
    sheet_name: str | None = None
    table_name: str | None = None
    data_start_row: int | None = None


class TableModel(BaseModel):
    """Base class for Pandas-backed tables with optional schema validation.

    This helper wires Pydantic-based metadata together with Pandera schemas to
    normalise ``DataFrame`` inputs, enforce required columns, manage default
    values, and provide a shared set of conversion helpers. Subclasses typically
    configure the table name, required columns, default values, an optional
    Pandera schema, and—when needed—the columns that form the dataframe index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    table_name: ClassVar[str] = ""
    required_columns: ClassVar[tuple[str, ...]] = ()
    default_values: ClassVar[dict[str, object]] = {}
    schema_model: ClassVar[type[pa.DataFrameModel] | None] = None
    index_columns: ClassVar[tuple[str, ...]] = ()

    df: pd.DataFrame = Field(default_factory=pd.DataFrame, repr=False)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        context: TableLoadContext | None = None,
    ) -> Self:
        """
        Create a table from a raw dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe that should contain the configured required columns.
        context : TableLoadContext | None, optional
            Optional metadata about workbook/sheet/table origin used to enrich
            validation errors with source locations.

        Returns
        -------
        Self
            Instance populated with a validated dataframe, indexed according to
            ``index_columns`` when provided.

        Raises
        ------
        ValueError
            If required columns are missing and no default values are available.
        """
        df_copy = df.copy()
        missing = []
        for col in cls.required_columns:
            if col not in df_copy.columns and col not in cls.default_values:
                missing.append(col)

        if missing:
            message = f"{cls.table_name} table missing columns: {sorted(missing)}"
            raise ValueError(cls._format_with_context(message, context))

        for col, default in cls.default_values.items():
            if col not in df_copy.columns:
                df_copy[col] = default

        # ensure column ordering starts with required columns if present
        column_order = [col for col in cls.required_columns if col in df_copy.columns]
        rest = [col for col in df_copy.columns if col not in column_order]
        ordered = df_copy[column_order + rest] if column_order else df_copy

        if cls.schema_model is not None:
            try:
                validated_df = cls.schema_model.validate(ordered)
            except (SchemaError, SchemaErrors) as exc:
                raise ValueError(cls._build_schema_error_message(exc, context)) from exc
        else:
            validated_df = ordered

        if cls.index_columns and all(col in validated_df.columns for col in cls.index_columns):
            validated_df = validated_df.set_index(list(cls.index_columns), drop=True)
        else:
            validated_df = validated_df.reset_index(drop=True)

        return cls(df=validated_df)

    @classmethod
    def _format_with_context(cls, message: str, context: TableLoadContext | None) -> str:
        """Append workbook/sheet/table details to an error message when available."""
        if context is None:
            return message

        details: list[str] = []
        if context.workbook_path is not None:
            details.append(f"workbook={Path(context.workbook_path)}")
        if context.sheet_name:
            details.append(f"sheet={context.sheet_name!r}")
        table_name = context.table_name or cls.table_name or cls.__name__
        details.append(f"table={table_name!r}")
        if context.data_start_row is not None:
            details.append(f"data_start_row={context.data_start_row}")

        return f"{message} ({', '.join(details)})"

    @staticmethod
    def _extract_failure_indices(error: SchemaError | SchemaErrors) -> list[int]:
        """Return distinct dataframe row indices reported by Pandera."""
        failure_cases = getattr(error, "failure_cases", None)
        if not isinstance(failure_cases, pd.DataFrame) or "index" not in failure_cases.columns:
            return []

        row_indices: set[int] = set()
        for value in failure_cases["index"].dropna().tolist():
            parsed = TableModel._to_int_index(value)
            if parsed is not None and parsed >= 0:
                row_indices.add(parsed)

        return sorted(row_indices)

    @staticmethod
    def _to_int_index(value: object) -> int | None:
        """Convert a Pandera failure-case index value to an integer when possible."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        if isinstance(value, Real):
            value_float = float(value)
            if value_float.is_integer():
                return int(value_float)
        return None

    @classmethod
    def _build_schema_error_message(
        cls,
        error: SchemaError | SchemaErrors,
        context: TableLoadContext | None,
    ) -> str:
        """Create a descriptive validation failure message from Pandera exceptions."""
        parts: list[str] = []

        column_name = getattr(error, "column_name", None)
        if isinstance(column_name, str):
            parts.append(f"column={column_name!r}")

        reason_code = getattr(error, "reason_code", None)
        if reason_code is not None:
            parts.append(f"reason={getattr(reason_code, 'value', str(reason_code))!r}")

        row_indices = cls._extract_failure_indices(error)
        if row_indices:
            parts.append(f"dataframe_rows={row_indices}")
            if context is not None and context.data_start_row is not None:
                excel_rows = [context.data_start_row + row for row in row_indices]
                parts.append(f"excel_rows={excel_rows}")

        details = f" [{'; '.join(parts)}]" if parts else ""
        base = f"Data validation failed: {error}"
        return cls._format_with_context(f"{base}{details}", context)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a defensive copy of the stored dataframe.

        Returns
        -------
        pd.DataFrame
            Copy of the current table contents with required columns preserved.
        """
        if self.df.empty:
            return pd.DataFrame(columns=list(self.required_columns) or [])
        return self.df.copy()

    def to_records(self) -> pd.DataFrame:
        """
        Return a dataframe copy with any configured index columns restored.

        This is primarily used by the SQLite persistence layer to export table
        contents without losing multi-index columns such as ``Faalpad_ID`` and
        ``Knoop_ID``.
        """
        df_repr = self.to_dataframe()
        if self.index_columns:
            return df_repr.reset_index()
        return df_repr

    def to_required_records(self) -> pd.DataFrame:
        """
        Return only the required columns as defined by ``required_columns``.

        Returns
        -------
        pd.DataFrame
            Subset of :meth:`to_records` limited to the required columns.

        Raises
        ------
        ValueError
            If one or more required columns are missing from the dataframe.
        """
        df_repr = self.to_records()
        if not self.required_columns:
            return df_repr
        missing = [col for col in self.required_columns if col not in df_repr.columns]
        if missing:
            raise ValueError(f"Required columns missing from dataframe: {missing}")
        return df_repr.loc[:, list(self.required_columns)]

    @classmethod
    def from_records(
        cls,
        df: pd.DataFrame,
        *,
        context: TableLoadContext | None = None,
    ) -> Self:
        """
        Construct an instance from a normalized dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe typically originating from SQLite or another serialized
            store.
        context : TableLoadContext | None, optional
            Optional metadata about workbook/sheet/table origin used to enrich
            validation errors with source locations.

        Returns
        -------
        Self
            Validated table containing the provided records.
        """
        return cls.from_dataframe(df, context=context)

    def __repr__(self) -> str:
        """Represent the table with its dataframe contents."""
        df_repr = self.to_dataframe()
        label = self.table_name or self.__class__.__name__
        return f"{label}\n{df_repr.__repr__()}"

    def _repr_html_(self) -> str:
        """Render the table as HTML for notebook-friendly display."""
        df_repr = self.to_dataframe()
        label = self.table_name or self.__class__.__name__
        return df_repr._repr_html_().replace("<table", f"<table data-label='{label}'", 1)
