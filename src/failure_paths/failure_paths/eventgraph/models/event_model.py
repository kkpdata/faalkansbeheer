from __future__ import annotations

from typing import ClassVar

import numpy as np
import openturns as ot
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from ...common.interp import interpolate_beta_curve
from ...common.prob import (
    INTERPOLATION_BETA_CAP,
    INTERPOLATION_PROB_EPSILON,
    beta_from_pf,
    pf_from_beta,
)
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
    std_normal: ClassVar[ot.Normal] = ot.Normal()
    interpolation_prob_epsilon: ClassVar[float] = INTERPOLATION_PROB_EPSILON
    interpolation_beta_cap: ClassVar[float] = INTERPOLATION_BETA_CAP

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
        # Use [node_id] to force a dataframe, even if only a single row is returned
        subset = self.df.loc[[node_id], ["h", "Beta_h"]].sort_values(["h", "Beta_h"], ascending=[True, False])

        # keep only the last +inf row, drop the rest
        inf_mask = np.isposinf(subset["Beta_h"])
        keep_mask = ~inf_mask | (inf_mask & (inf_mask.cumsum() == inf_mask.sum()))
        subset = subset[keep_mask]

        # keep only the first -inf row, drop the rest
        ninf_mask = np.isneginf(subset["Beta_h"])
        keep_mask = ~ninf_mask | (ninf_mask & (ninf_mask.cumsum() == 1))
        subset = subset[keep_mask]

        h_array = np.asarray(h, dtype=float)
        if len(subset) == 1:
            beta_vals = np.full(h_array.shape, subset.Beta_h.iat[0])
        else:
            h_nodes = subset.h.to_numpy()
            beta_nodes = subset.Beta_h.to_numpy()
            beta_vals = interpolate_beta_curve(
                h_nodes,
                beta_nodes,
                h_array,
                beta_cap=float(self.interpolation_beta_cap),
                preserve_exact_knots=True,
                clamp_infinite_tails=True,
            )

        if as_beta:
            return beta_vals

        return pf_from_beta(beta_vals, tail="upper")

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
        values = pf.loc[mask].astype(float).to_numpy()
        beta_filled = beta.astype(float)
        beta_filled.loc[mask] = beta_from_pf(values, tail="upper")

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
        values = beta.loc[mask].astype(float).to_numpy(dtype=float)
        pf_filled = pf.astype(float)
        pf_filled.loc[mask] = pf_from_beta(values, tail="upper")
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

        # Drop exact duplicates
        table.df = table.df.reset_index().drop_duplicates(subset=list(cls.required_columns))
        table.df = table.df.set_index(list(cls.index_columns))

        # Validate unique combinations of Faalpad_ID and Knoop_ID
        dupe_subset = ["Faalpad_ID", "Knoop_ID", "h"]
        duplicated = table.df.reset_index().duplicated(subset=dupe_subset)
        if duplicated.any():
            duplicates = table.df.reset_index().loc[duplicated, dupe_subset]
            dup_str = duplicates.to_dict(orient="records").__repr__()
            raise ValueError(f"Duplicate events detected for combinations of {dupe_subset}: {dup_str}")

        # Validate monotone fragility curves per event id.
        # Regular faalpaden must be non-decreasing with h.
        # Faalpaden -1 and -2 are synthetic/special and must be monotone:
        # entirely non-decreasing or entirely non-increasing.
        events_df = table.df.reset_index().sort_values(["Faalpad_ID", "Knoop_ID", "h"])
        violations: list[dict[str, float | int]] = []
        for (faalpad_id, knoop_id), group in events_df.groupby(["Faalpad_ID", "Knoop_ID"], sort=False):
            h_values = group["h"].to_numpy(dtype=float)
            pf_values = group["Pf_h"].to_numpy(dtype=float)
            diff_pf = np.diff(pf_values)
            decreasing = diff_pf < 0.0
            increasing = diff_pf > 0.0
            is_special = int(faalpad_id) in {-1, -2}

            if is_special:
                # Mixed direction is invalid for special ids.
                if not (np.any(decreasing) and np.any(increasing)):
                    continue
                offenders = np.where(decreasing | increasing)[0]
            else:
                if not np.any(decreasing):
                    continue
                offenders = np.where(decreasing)[0]

            for idx in offenders:
                violations.append(
                    {
                        "Faalpad_ID": int(faalpad_id),
                        "Knoop_ID": int(knoop_id),
                        "h_prev": float(h_values[idx]),
                        "Pf_prev": float(pf_values[idx]),
                        "h_next": float(h_values[idx + 1]),
                        "Pf_next": float(pf_values[idx + 1]),
                    }
                )

        if violations:
            sample = violations[:5]
            raise ValueError(
                "Invalid fragility curve: Pf_h must be non-decreasing with increasing h; "
                "special Faalpad_ID -1/-2 must be monotone. "
                f"Violations (showing up to 5): {sample}"
            )

        return table
