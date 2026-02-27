from __future__ import annotations

from typing import ClassVar

import pandera.pandas as pa
from pandera.typing import Series

from .table_model import TableModel


class MetadataSchema(pa.DataFrameModel):
    """Schema describing workbook-level metadata."""

    TRAJECT_ID: Series[str] = pa.Field(coerce=True)
    M_VAN: Series[float] = pa.Field(coerce=True)
    M_TOT: Series[float] = pa.Field(coerce=True)
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
        "TRAJECT_ID",
        "M_VAN",
        "M_TOT",
        "dijkvaknummer",
        "Vaknaam",
        "LENGTE_VAK",
        "TYPE_WATERKERING",
        "Ondergrondscenario",
        "ScenarioKans",
        "HR_locatie",
    )
    schema_model: ClassVar[type[pa.DataFrameModel]] = MetadataSchema
