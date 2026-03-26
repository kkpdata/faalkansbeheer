from __future__ import annotations

from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from .table_model import TableLoadContext, TableModel


class FrequencySchema(pa.DataFrameModel):
    """Schema describing water-level exceedance frequencies."""

    h: Series[float] = pa.Field(coerce=True)
    Pf_h: Series[float] = pa.Field(coerce=True)


class FrequencyTable(TableModel):
    """Store computed exceedance frequencies as a validated table."""

    table_name: ClassVar[str] = "hfreq"
    required_columns: ClassVar[tuple[str, ...]] = ("h", "Pf_h")
    schema_model: ClassVar[type[pa.DataFrameModel]] = FrequencySchema

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        context: TableLoadContext | None = None,
    ) -> FrequencyTable:
        """Create a frequency table while discarding fully empty required rows."""
        df_copy = df.copy()
        return super().from_dataframe(df_copy, context=context)
