from __future__ import annotations

from typing import ClassVar

import pandera.pandas as pa
from pandera.typing import Series

from .table_model import TableModel


class MetadataSchema(pa.DataFrameModel):
    """Schema describing workbook-level metadata."""

    dijkvaknummer: Series[int] = pa.Field(coerce=True)
    Vaknaam: Series[str] = pa.Field(coerce=True)
    LENGTE_VAK: Series[float] = pa.Field(coerce=True)
    TYPE_WATERKERING: Series[str] = pa.Field(coerce=True)
    Ondergrondscenario: Series[str] = pa.Field(coerce=True)
    ScenarioKans: Series[float] = pa.Field(coerce=True)
    HR_locatie: Series[str] = pa.Field(coerce=True)


class MetadataTable(TableModel):
    """Store workbook metadata as a validated single-row table."""

    table_name: ClassVar[str] = "metadata"
    required_columns: ClassVar[tuple[str, ...]] = (
        "dijkvaknummer",
        "Vaknaam",
        "LENGTE_VAK",
        "TYPE_WATERKERING",
        "Ondergrondscenario",
        "ScenarioKans",
        "HR_locatie",
    )
    schema_model: ClassVar[type[pa.DataFrameModel]] = MetadataSchema
