from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from scipy.stats import norm

from ...common.interp import LinearInterpolator
from .table_model import TableModel


class EventSchema(pa.DataFrameModel):
    """Schema describing event-level failure probabilities."""

    Faalpad_ID: Series[int] = pa.Field(coerce=True)
    Knoop_ID: Series[int] = pa.Field(coerce=True)
    h: Series[float] = pa.Field(coerce=True)
    Pf_h: Series[float] = pa.Field(coerce=True, nullable=True)
    Beta_h: Series[float] = pa.Field(coerce=True, nullable=True)


class EventTable(TableModel):
    """Store failure probabilities per event along a path."""

    table_name: ClassVar[str] = "faalpadkansen"
    required_columns: ClassVar[tuple[str, ...]] = (
        "Faalpad_ID",
        "Knoop_ID",
        "h",
        "Pf_h",
        "Beta_h",
    )
    schema_model: ClassVar[type[pa.DataFrameModel]] = EventSchema
    index_columns: ClassVar[tuple[str, ...]] = ("Faalpad_ID", "Knoop_ID")

    def get_event_ids(self) -> set[tuple[int, int]]:
        """Return the distinct ``(Faalpad_ID, Knoop_ID)`` combinations in the table."""
        return set(self.df.index)

    def get_event_prob(
        self,
        node_id: tuple[int, int],
        h: float,
        as_beta: bool = False,
    ) -> float:
        """
        Interpolate a single water level for the requested node.

        Parameters
        ----------
        node_id : tuple[int, int]
            ``(Faalpad_ID, Knoop_ID)`` tuple that identifies the event.
        h : float
            Water level used for interpolation.
        as_beta : bool
            When ``True``, return reliability indices instead of ``Pf`` values.

        Returns
        -------
        float
            Interpolated probability or reliability index.
        """
        return self.get_event_probs(node_id, [h], as_beta=as_beta)[0]

    def get_event_probs(
        self,
        node_id: tuple[int, int],
        h: np.ndarray | list[float],
        as_beta: bool = False,
    ) -> np.ndarray:
        """
        Interpolate reliability indices or Pf values for ``node_id`` at the provided water levels.

        Parameters
        ----------
        node_id : tuple[int, int]
            ``(Faalpad_ID, Knoop_ID)`` tuple that identifies the event.
        h : np.ndarray | list[float]
            Water levels used for interpolation.
        as_beta : bool
            When ``True``, return reliability indices instead of ``Pf`` values.

        Returns
        -------
        numpy.ndarray
            Interpolated probabilities or reliability indices for each water level.
        """
        subset = self.df.loc[node_id, ["h", "Beta_h"]]
        h_array = np.asarray(h, dtype=float)
        if len(subset) == 1:
            interp_vals = np.full(h_array.shape, subset.Beta_h.iat[0])
        else:
            interpolator = LinearInterpolator(subset.h.to_numpy(), subset.Beta_h.to_numpy())
            interp_vals = interpolator.value(h_array)

        if not as_beta:
            interp_vals = norm.sf(interp_vals)

        return interp_vals

    @classmethod
    def _fill_beta_from_pf(cls, pf: pd.Series, beta: pd.Series) -> pd.Series:
        """Fill missing ``Beta_h`` entries using available ``Pf_h`` values.

        Parameters
        ----------
        pf : pd.Series
            Column containing failure probabilities, possibly with ``NaN``.
        beta : pd.Series
            Reliability index column to augment.

        Returns
        -------
        pd.Series
            Updated ``Beta_h`` column with derived entries populated.
        """
        mask = beta.isna()
        if not mask.any():
            return beta
        values = pf.loc[mask].astype(float)
        beta_filled = beta.astype(float)
        beta_filled.loc[mask] = norm.isf(values)

        # Replace inf values
        beta_filled.loc[np.isposinf(beta_filled)] = 9999999.9
        beta_filled.loc[np.isneginf(beta_filled)] = -9999999.9

        return beta_filled

    @classmethod
    def _fill_pf_from_beta(cls, pf: pd.Series, beta: pd.Series) -> pd.Series:
        """Fill missing ``Pf_h`` entries using available ``Beta_h`` values.

        Parameters
        ----------
        pf : pd.Series
            Column containing failure probabilities to augment.
        beta : pd.Series
            Column containing reliability indices (may include ``NaN``).

        Returns
        -------
        pd.Series
            Updated ``Pf_h`` column with derived entries populated.
        """
        mask = pf.isna()
        if not mask.any():
            return pf
        values = beta.loc[mask].astype(float)
        pf_filled = pf.astype(float)
        pf_filled.loc[mask] = norm.sf(values)
        return pf_filled

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> EventTable:
        """Create an :class:`EventTable` with Pf/Beta completeness safeguards.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with event probability information.

        Returns
        -------
        EventTable
            Validated table with derived Pf/Beta values when missing.

        Raises
        ------
        ValueError
            If any row lacks both ``Pf_h`` and ``Beta_h`` values.
        """
        table = super().from_dataframe(df)
        table.df = table.df.sort_index()
        pf = table.df["Pf_h"]
        beta = table.df["Beta_h"]

        # Validate that at least Pf_h or Beta_h is defined for each row
        missing_both = pf.isna() & beta.isna()
        if missing_both.any():
            missing_rows = table.df.index[missing_both].tolist()
            raise ValueError(f"Each row must define Pf_h or Beta_h; missing rows: {missing_rows}")

        # Fill beta and pf value
        table.df["Beta_h"] = cls._fill_beta_from_pf(pf, beta)
        table.df["Pf_h"] = cls._fill_pf_from_beta(pf, beta)

        # Validate unique combinations of Faalpad_ID and Knoop_ID
        dupe_subset = ["Faalpad_ID", "Knoop_ID", "h"]
        duplicated = table.df.reset_index().duplicated(subset=dupe_subset)
        if duplicated.any():
            duplicates = table.df.reset_index().loc[duplicated, dupe_subset]
            dup_str = duplicates.to_dict(orient="records").__repr__()
            raise ValueError(f"Duplicate events detected for combinations of {dupe_subset}: {dup_str}")

        return table
