from __future__ import annotations

from typing import ClassVar, Self

import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, ConfigDict, Field


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
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        """
        Create a table from a raw dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe that should contain the configured required columns.

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
            raise ValueError(f"{cls.table_name} table missing columns: {sorted(missing)}")

        for col, default in cls.default_values.items():
            if col not in df_copy.columns:
                df_copy[col] = default

        # ensure column ordering starts with required columns if present
        column_order = [col for col in cls.required_columns if col in df_copy.columns]
        rest = [col for col in df_copy.columns if col not in column_order]
        ordered = df_copy[column_order + rest] if column_order else df_copy

        if cls.schema_model is not None:
            validated_df = cls.schema_model.validate(ordered)
        else:
            validated_df = ordered

        if cls.index_columns and all(col in validated_df.columns for col in cls.index_columns):
            validated_df = validated_df.set_index(list(cls.index_columns), drop=True)
        else:
            validated_df = validated_df.reset_index(drop=True)

        return cls(df=validated_df)

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
    def from_records(cls, df: pd.DataFrame) -> Self:
        """
        Construct an instance from a normalized dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe typically originating from SQLite or another serialized
            store.

        Returns
        -------
        Self
            Validated table containing the provided records.
        """
        return cls.from_dataframe(df)

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
