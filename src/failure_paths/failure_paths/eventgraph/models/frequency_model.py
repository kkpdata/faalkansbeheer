from __future__ import annotations

from typing import ClassVar

import pandera.pandas as pa
from pandera.typing import Series

from .table_model import TableModel


class FrequencySchema(pa.DataFrameModel):
    """Schema describing water-level exceedance frequencies."""

    h: Series[float] = pa.Field(coerce=True)
    Pf_h: Series[float] = pa.Field(coerce=True)


class FrequencyTable(TableModel):
    """Store computed exceedance frequencies as a validated table."""

    table_name: ClassVar[str] = "hfreq"
    required_columns: ClassVar[tuple[str, ...]] = ("h", "Pf_h")
    schema_model: ClassVar[type[pa.DataFrameModel]] = FrequencySchema
