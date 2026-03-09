from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class LinearInterpolator:
    """
    Lightweight wrapper around :func:`scipy.interpolate.interp1d`.

    Parameters
    ----------
    x : np.ndarray
        Knot locations (e.g. water levels). The array may be unsorted and may contain duplicates.
    y : np.ndarray
        Function values at the knots. Must have the same length as ``x``.

    Notes
    -----
    * Inputs are copied and flattened, but sorting is assumed to be done by the
      caller
    * Duplicate ``x`` values (plateaus) are preserved and therefore still
      map to the *last* ``y`` value at that level when querying exactly on
      the plateau. This is achieved by composing two SciPy interpolants:
      one bounded spline that mirrors plateau semantics (returns ``NaN`` outside
      the range) and a lazy extrapolating spline used only to fill those ``NaN``
      entries. As a result, evaluations outside the knot range still follow
      the linear continuation.

    Raises
    ------
    ValueError
        Raised when ``x``/``y`` lengths differ, fewer than two points are provided, or the
        sorted ``x`` knots are decreasing.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size != y_arr.size:
            raise ValueError("x and y must have identical lengths.")
        if x_arr.size < 2:
            raise ValueError("At least two points are required for interpolation.")

        if np.any(np.diff(x_arr) < 0):
            raise ValueError("x values must be non-decreasing.")

        self._x_nodes = x_arr
        self._y_nodes = y_arr
        self._interp = interp1d(
            x_arr,
            y_arr,
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=False,
        )
        self._extrap = interp1d(
            x_arr,
            y_arr,
            fill_value="extrapolate",
            assume_sorted=False,
        )

    def value(self, x_query: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the interpolant at new points.

        Parameters
        ----------
        x_query : np.ndarray | float
            Scalar or array-like coordinates at which to sample the interpolant.

        Returns
        -------
        np.ndarray | float
            Interpolated values with the same shape as ``x_query``.
        """
        is_scalar = np.isscalar(x_query)
        query = np.asarray(x_query, dtype=float)
        with np.errstate(divide="ignore"):
            result = self._interp(query)

        if is_scalar:
            value = float(result)
            if np.isnan(value):
                with np.errstate(divide="ignore"):
                    return float(self._extrap(query))
            return value

        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            with np.errstate(divide="ignore"):
                result[nan_mask] = self._extrap(query[nan_mask])

        if is_scalar:
            return float(result)
        return result

    def inverse(self, y_query: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the inverse mapping ``y -> x`` using the same semantics as :meth:`value`.

        Parameters
        ----------
        y_query : np.ndarray | float
            Scalar or array-like ``y`` values whose corresponding ``x`` knots
            should be interpolated.

        Returns
        -------
        np.ndarray | float
            Interpolated ``x`` values.
        """
        # @TODO
        inverse_interp = LinearInterpolator(self._y_nodes, self._x_nodes)
        return inverse_interp.value(y_query)


def interpolate_beta_curve(
    x_nodes: np.ndarray | list[float],
    beta_nodes: np.ndarray | list[float],
    x_query: np.ndarray | float,
    *,
    beta_cap: float,
    preserve_exact_knots: bool = True,
    clamp_infinite_tails: bool = True,
) -> np.ndarray | float:
    """
    Interpolate beta values robustly when knots may include +/-inf.

    Parameters
    ----------
    x_nodes : np.ndarray | list[float]
        Knot locations. Must be non-decreasing.
    beta_nodes : np.ndarray | list[float]
        Beta values at knot locations. May include +/-inf.
    x_query : np.ndarray | float
        Query coordinates where beta should be interpolated.
    beta_cap : float
        Finite cap used to replace +/-inf during interpolation.
    preserve_exact_knots : bool, optional
        Restore exact knot beta values after interpolation.
    clamp_infinite_tails : bool, optional
        Clamp outside-domain values to endpoint beta when the endpoint is infinite.

    Returns
    -------
    np.ndarray | float
        Interpolated beta values with the same shape as ``x_query``.

    Raises
    ------
    ValueError
        If ``x_nodes`` and ``beta_nodes`` lengths differ, or no knots are provided.
    """
    x_arr = np.asarray(x_nodes, dtype=float).reshape(-1)
    b_arr = np.asarray(beta_nodes, dtype=float).reshape(-1)
    if x_arr.size != b_arr.size:
        raise ValueError("x_nodes and beta_nodes must have identical lengths.")
    if x_arr.size == 0:
        raise ValueError("At least one knot is required.")

    is_scalar = np.isscalar(x_query)
    q_arr = np.asarray(x_query, dtype=float)
    q_flat = q_arr.reshape(-1)

    if x_arr.size == 1:
        out_flat = np.full(q_flat.shape, b_arr[0], dtype=float)
    else:
        interp_nodes = np.nan_to_num(b_arr, posinf=float(beta_cap), neginf=-float(beta_cap))
        out_flat = np.asarray(LinearInterpolator(x_arr, interp_nodes).value(q_flat), dtype=float).reshape(-1)

        if preserve_exact_knots:
            idx = np.searchsorted(x_arr, q_flat, side="right") - 1
            valid = idx >= 0
            if np.any(valid):
                positions = np.nonzero(valid)[0]
                idx_valid = idx[valid]
                exact = x_arr[idx_valid] == q_flat[valid]
                out_flat[positions[exact]] = b_arr[idx_valid[exact]]

        if clamp_infinite_tails:
            if np.isinf(b_arr[0]):
                out_flat[q_flat < x_arr[0]] = b_arr[0]
            if np.isinf(b_arr[-1]):
                out_flat[q_flat > x_arr[-1]] = b_arr[-1]

    out = out_flat.reshape(q_arr.shape)
    if is_scalar:
        return float(out.reshape(-1)[0])
    return out
