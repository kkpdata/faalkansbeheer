from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class LinearInterpolator:
    """
    Lightweight wrapper around :func:`scipy.interpolate.interp1d`.

    Notes
    -----
    * Inputs are copied, flattened, and stably sorted so callers may pass
      unsorted knots without caring about their original ordering.
    * Duplicate ``x`` values (plateaus) are preserved and therefore still
      map to the *last* ``y`` value at that level when querying exactly on
      the plateau; SciPy's own extrapolation path handles this once the
      data stay sorted.
    * ``interp1d`` is instantiated with ``fill_value="extrapolate"`` so
      evaluations outside the knot range stay linear.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Create a 1D linear interpolator on the provided knots.

        Parameters
        ----------
        x :
            Knot locations (e.g. water levels). The array may be unsorted and
            may contain duplicate values.
        y :
            Function values at the knots. Must have the same length as ``x``.

        Raises
        ------
        ValueError
            If ``x``/``y`` lengths differ, fewer than two points are provided,
            or the sorted ``x`` knots are decreasing (e.g. ``[2, 1, 0]``).
        """
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size != y_arr.size:
            raise ValueError("x and y must have identical lengths.")
        if x_arr.size < 2:
            raise ValueError("At least two points are required for interpolation.")

        order = np.argsort(x_arr, kind="mergesort")
        x_arr = x_arr[order]
        y_arr = y_arr[order]

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
        self._extrap = interp1d(x_arr, y_arr, fill_value="extrapolate", assume_sorted=False)

    def value(self, x_query: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the interpolant at new points.

        Parameters
        ----------
        x_query :
            Scalar or array-like coordinates at which to sample the interpolant.

        Returns
        -------
        numpy.ndarray | float
            Interpolated values with the same shape as ``x_query``.
        """
        is_scalar = np.isscalar(x_query)
        query = np.asarray(x_query, dtype=float)
        result = self._interp(query)
        if is_scalar:
            value = float(result)
            if np.isnan(value):
                return float(self._extrap(query))
            return value

        nan_mask = np.isnan(result)
        if np.any(nan_mask):
            result[nan_mask] = self._extrap(query[nan_mask])

        if is_scalar:
            return float(result)
        return result

    def inverse(self, y_query: np.ndarray | float) -> np.ndarray | float:
        """
        Evaluate the inverse mapping ``y -> x`` using the same semantics as :meth:`value`.

        Parameters
        ----------
        y_query :
            Scalar or array-like ``y`` values whose corresponding ``x`` knots
            should be interpolated.

        Returns
        -------
        numpy.ndarray | float
            Interpolated ``x`` values.
        """
        inverse_interp = LinearInterpolator(self._y_nodes, self._x_nodes)
        return inverse_interp.value(y_query)
