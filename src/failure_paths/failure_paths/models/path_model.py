from __future__ import annotations

import re
from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from .table_model import TableModel


class PathSchema(pa.DataFrameModel):
    """Schema describing the base columns of the failure-path table."""

    Faalpad_ID: Series[int] = pa.Field(coerce=True, ge=1)
    Overslag: Series[str] = pa.Field(coerce=True, isin=["ja", "nee"])
    Indirect_mechanisme: Series[str] = pa.Field(coerce=True)
    Initiele_gebeurtenis: Series[str] = pa.Field(coerce=True)
    Vervolggebeurtenis_1: Series[str] = pa.Field(coerce=True)
    Vervolggebeurtenis_2: Series[str] = pa.Field(coerce=True, nullable=True)
    Vervolggebeurtenis_3: Series[str] = pa.Field(coerce=True, nullable=True)
    Vervolggebeurtenis_4: Series[str] = pa.Field(coerce=True, nullable=True)
    Vervolggebeurtenis_5: Series[str] = pa.Field(coerce=True, nullable=True)
    Vervolggebeurtenis_6: Series[str] = pa.Field(coerce=True, nullable=True)
    Vervolggebeurtenis_7: Series[str] = pa.Field(coerce=True, nullable=True)

    class Config:
        strict = False


class PathTable(TableModel):
    """Table wrapper that stores a graph of failure paths."""

    table_name: ClassVar[str] = "faalpadschema"
    required_columns: ClassVar[tuple[str, ...]] = (
        "Faalpad_ID",
        "Overslag",
        "Indirect_mechanisme",
        "Initiele_gebeurtenis",
        "Vervolggebeurtenis_1",
        "Vervolggebeurtenis_2",
        "Vervolggebeurtenis_3",
        "Vervolggebeurtenis_4",
        "Vervolggebeurtenis_5",
        "Vervolggebeurtenis_6",
        "Vervolggebeurtenis_7",
    )
    schema_model: ClassVar[type[pa.DataFrameModel]] = PathSchema
    _schema_cache: ClassVar[dict[tuple[str, ...], type[pa.DataFrameModel]]] = {}
    followup_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^Vervolggebeurtenis_(?P<index>[1-9]\d*)$")

    @staticmethod
    def _normalize_str(value: object) -> object:
        """
        Return a stripped and lower-cased string while leaving other values untouched.

        Parameters
        ----------
        value : object
            Input value taken from the ``Overslag`` column.

        Returns
        -------
        object
            Lower-cased stripped string when ``value`` is a ``str``; the original value otherwise.
        """
        return value.strip().lower() if isinstance(value, str) else value

    @staticmethod
    def _validate_indirect_mechanism(series: pd.Series) -> None:
        """Validate the allowed unique values for the ``Indirect_mechanisme`` column.

        Parameters
        ----------
        series : pd.Series
            Normalized ``Indirect_mechanisme`` column that should contain between
            one and two unique string values.

        Raises
        ------
        ValueError
            If non-string values are present, if the column has zero unique
            values, if more than two unique values are detected, or if two
            distinct values are present without ``"nee"``.
        """
        unique_values = list(series.dropna().unique())
        non_str_values = [value for value in unique_values if not isinstance(value, str)]
        if non_str_values:
            raise ValueError(f"Indirect_mechanisme must contain strings only; invalid values: {non_str_values}")
        if len(unique_values) == 0:
            raise ValueError("Indirect_mechanisme must contain at most two unique values; found none")
        if len(unique_values) > 2:
            raise ValueError(f"Indirect_mechanisme must contain at most two unique values; found: {unique_values}")
        if len(unique_values) == 2 and "nee" not in unique_values:
            raise ValueError(f"Indirect_mechanisme has two unique values but not 'nee'; found: {unique_values}")

    @classmethod
    def _schema_for_columns(cls, columns: pd.Index) -> type[pa.DataFrameModel]:
        """Return a schema that includes every detected follow-up column.

        Parameters
        ----------
        columns : pd.Index
            Incoming dataframe columns.

        Returns
        -------
        type[pa.DataFrameModel]
            The schema class capable of validating the detected columns.

        Raises
        ------
        ValueError
            If follow-up columns are missing the numeric suffix.
        """
        followup_cols = set()
        invalid_followups = []
        for column in columns:
            match = cls.followup_pattern.match(column)
            if match:
                followup_cols.add(column)
            elif column.startswith("Vervolggebeurtenis_"):
                invalid_followups.append(column)

        if invalid_followups:
            numbered = ", ".join(sorted(invalid_followups))
            raise ValueError(
                f"Follow-up columns must end with a positive integer suffix; invalid columns detected: {numbered}"
            )
        base_fields = getattr(PathSchema, "__annotations__", {})

        # Only generate new schema classes when columns beyond the static schema appear.
        extra_cols = tuple(sorted(followup_cols - base_fields.keys()))
        if not extra_cols:
            return PathSchema

        cached_schema = cls._schema_cache.get(extra_cols)
        if cached_schema:
            # Identical column sets reuse the same dynamically built schema.
            return cached_schema

        # Cache schema subclasses so identical column sets reuse the same class.
        namespace = {
            "__annotations__": dict.fromkeys(extra_cols, Series[str]),
            **{col: pa.Field(coerce=True, nullable=True) for col in extra_cols},
        }
        suffix = "_".join(extra_cols).replace("Vervolggebeurtenis_", "")
        schema_name = f"PathSchemaDynamic_{suffix}"
        dynamic_schema = type(schema_name, (PathSchema,), namespace)
        cls._schema_cache[extra_cols] = dynamic_schema

        return dynamic_schema

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> PathTable:
        """
        Build a :class:`PathTable` from a dataframe with arbitrary follow-ups.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing failure paths and follow-up events.

        Returns
        -------
        PathTable
            Validated table with normalized follow-up columns and overslag values.
        """
        # Drop rows that are nan except for the first column
        df_copy = df.copy()
        df_copy = df_copy.dropna(axis=0, how="all", subset=df_copy.columns.tolist()[1:])

        # Normalize overslag and indirect mechanism
        if "Overslag" in df.columns:
            df_copy["Overslag"] = df_copy["Overslag"].map(cls._normalize_str)
        if "Indirect_mechanisme" in df.columns:
            df_copy["Indirect_mechanisme"] = df_copy["Indirect_mechanisme"].map(cls._normalize_str)
            cls._validate_indirect_mechanism(df_copy["Indirect_mechanisme"])

        schema_cls = cls._schema_for_columns(df_copy.columns)
        previous_schema = cls.schema_model
        cls.schema_model = schema_cls
        try:
            return super().from_dataframe(df_copy)
        finally:
            # Always restore the base schema so subsequent calls start from a
            # clean state, regardless of validation success or failure.
            cls.schema_model = previous_schema
