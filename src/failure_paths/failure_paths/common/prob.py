#!/usr/bin/env python3
import numpy as np
import openturns as ot
from numpy.typing import NDArray


def clamp_probabilities(
    probs: np.ndarray | float,
    *,
    min_prob: float | None = None,
    max_prob: float | None = None,
) -> NDArray[np.float64]:
    """
    Clamp probabilities to an optional [min_prob, max_prob] interval.

    Parameters
    ----------
    probs : np.ndarray | float
        Probability values to clamp.
    min_prob : float | None, optional
        Lower bound. When ``None``, no lower clipping is applied.
    max_prob : float | None, optional
        Upper bound. When ``None``, no upper clipping is applied.

    Returns
    -------
    numpy.ndarray
        Clamped probabilities as a float array with the same shape as ``probs``.
    """
    arr = np.asarray(probs, dtype=float)
    lower = -np.inf if min_prob is None else float(min_prob)
    upper = np.inf if max_prob is None else float(max_prob)
    return np.clip(arr, lower, upper)


def log_pf_from_beta(
    beta: np.ndarray | float,
    *,
    tail: str = "upper",
) -> NDArray[np.float64]:
    """
    Compute log(Pf) from reliability index beta in a stable way.

    Parameters
    ----------
    beta : np.ndarray | float
        Reliability indices.
    tail : str, optional
        Tail convention: ``"upper"`` uses Pf = SF(beta), ``"lower"`` uses
        Pf = CDF(beta).

    Returns
    -------
    numpy.ndarray
        Log probabilities with the same shape as ``beta``.

    Raises
    ------
    ValueError
        If ``tail`` is not one of ``"upper"`` or ``"lower"``.
    """
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")

    arr = np.asarray(beta, dtype=float)
    flat = arr.reshape(-1)
    if tail == "upper":
        log_pf = np.vectorize(lambda x: ot.DistFunc.logpNormal(float(x), True), otypes=[float])(flat)
    else:
        log_pf = np.vectorize(lambda x: ot.DistFunc.logpNormal(float(-x), True), otypes=[float])(flat)
    return log_pf.reshape(arr.shape)


def beta_from_pf(
    pf: np.ndarray | float,
    *,
    tail: str = "upper",
    min_prob: float | None = None,
    max_prob: float | None = None,
    inf_substitute: float | None = None,
) -> NDArray[np.float64]:
    """
    Convert failure probabilities to reliability indices.

    Parameters
    ----------
    pf : np.ndarray | float
        Failure probabilities.
    tail : str, optional
        Tail convention: ``"upper"`` maps Pf to the upper tail
        (uses inverse survival), ``"lower"`` uses the CDF quantile.
    min_prob : float | None, optional
        Optional lower bound for clipping. When ``None``, no clipping is applied.
    max_prob : float | None, optional
        Optional upper bound for clipping. When ``None``, no clipping is applied.
    inf_substitute : float | None, optional
        If provided, replace ``+/-inf`` results with this magnitude.

    Returns
    -------
    numpy.ndarray
        Reliability indices with the same shape as ``pf``.

    Raises
    ------
    ValueError
        If ``tail`` is not one of ``"upper"`` or ``"lower"``.
    """
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")

    probs = clamp_probabilities(pf, min_prob=min_prob, max_prob=max_prob)
    zero_mask = probs == 0.0
    one_mask = probs == 1.0
    flat = probs.reshape(-1)
    normal = ot.Normal()
    if tail == "upper":
        betas = np.array(normal.computeQuantile(flat, True))
    else:
        betas = np.array(normal.computeQuantile(flat))
    betas = betas.reshape(probs.shape)

    if np.any(zero_mask) or np.any(one_mask):
        if tail == "upper":
            betas = np.where(zero_mask, np.inf, betas)
            betas = np.where(one_mask, -np.inf, betas)
        else:
            betas = np.where(zero_mask, -np.inf, betas)
            betas = np.where(one_mask, np.inf, betas)

    if inf_substitute is not None:
        betas = np.nan_to_num(betas, nan=0.0, posinf=inf_substitute, neginf=-inf_substitute)
    return betas


# Shared defaults for stable interpolation when true 0/1 probabilities appear
# in fragility curves and would otherwise map to +/-inf betas.
INTERPOLATION_PROB_EPSILON = 1e-300
INTERPOLATION_BETA_CAP = float(beta_from_pf(np.array([INTERPOLATION_PROB_EPSILON]), tail="upper")[0])


def pf_from_beta(
    beta: np.ndarray | float,
    *,
    tail: str = "upper",
    log_floor: float | None = None,
) -> NDArray[np.float64]:
    """
    Convert reliability indices to failure probabilities.

    Parameters
    ----------
    beta : np.ndarray | float
        Reliability indices.
    tail : str, optional
        Tail convention: ``"upper"`` uses survival, ``"lower"`` uses CDF.
    log_floor : float | None, optional
        Optional floor applied to log(Pf) before exponentiation. When ``None``,
        no floor is applied and true zeros/ones are preserved.

    Returns
    -------
    numpy.ndarray
        Failure probabilities with the same shape as ``beta``.

    Raises
    ------
    ValueError
        If ``tail`` is not one of ``"upper"`` or ``"lower"``.
    """
    if tail not in {"upper", "lower"}:
        raise ValueError("tail must be 'upper' or 'lower'.")

    arr = np.asarray(beta, dtype=float)
    flat = arr.reshape(-1)
    neg_inf_mask = np.isneginf(flat)
    pos_inf_mask = np.isposinf(flat)
    flat_col = flat[:, np.newaxis]
    normal = ot.Normal()
    if tail == "upper":
        pf_vals = np.array(normal.computeSurvivalFunction(flat_col)).reshape(-1)
    else:
        pf_vals = np.array(normal.computeCDF(flat_col)).reshape(-1)

    if np.any(neg_inf_mask) or np.any(pos_inf_mask):
        if tail == "upper":
            pf_vals = np.where(neg_inf_mask, 1.0, pf_vals)
            pf_vals = np.where(pos_inf_mask, 0.0, pf_vals)
        else:
            pf_vals = np.where(neg_inf_mask, 0.0, pf_vals)
            pf_vals = np.where(pos_inf_mask, 1.0, pf_vals)

    if log_floor is not None:
        log_pf = np.log(pf_vals, where=pf_vals > 0, out=np.full_like(pf_vals, -np.inf))
        log_pf = np.maximum(log_pf, float(log_floor))
        pf_vals = np.exp(log_pf)

    return pf_vals.reshape(arr.shape)


def cumulative_beta_equivalent_ot(
    beta_matrix: np.ndarray | list[list[float]],
    axis: int = 0,
    log_floor: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute cumulative product of Pf = Phi(-beta) along an axis and return equivalent beta.

    Parameters
    ----------
    beta_matrix : np.ndarray | list[list[float]]
        Reliability indices (betas). Converted to a float array.
    axis : int, optional
        Axis to accumulate along: 0 accumulates down rows (within each column),
        1 accumulates across columns (within each row).
    log_floor : float | None, optional
        Minimum log probability; values below this are floored (Pf approx 1e-300).
        When ``None``, defaults to ``log(1e-300)``. For fold-to-zero behavior,
        set Pf to 0.0 when logPf < log_floor.

    Returns
    -------
    beta_cum : ndarray, shape (m, n)
        Equivalent cumulative beta.
    logPf_cum : ndarray, shape (m, n)
        Cumulative log(product Pf) along the axis.
    Pf_cum : ndarray, shape (m, n)
        Cumulative product Pf after flooring.

    Raises
    ------
    ValueError
        If `beta_matrix` is not 2D.
    """
    B = np.asarray(beta_matrix, dtype=float)
    if B.ndim != 2:
        raise ValueError("beta_matrix must be 2D.")

    # logPf = log(Phi(-beta)) = log(SF(beta)) for standard normal
    # OpenTURNS: logpNormal(x, tail=True) gives log complementary CDF (log SF).
    with np.errstate(divide="ignore"):
        logPf = np.vectorize(lambda x: ot.DistFunc.logpNormal(float(x), True), otypes=[float])(B)

    # cumulative log-products
    logPf_cum = np.cumsum(logPf, axis=axis)

    # Apply your underflow policy.
    # Option A (default): floor at log(1e-300) so Pf never goes below ~1e-300.
    if log_floor is None:
        log_floor = float(np.log(1e-300))
    logPf_cum_adj = np.maximum(logPf_cum, log_floor)
    Pf_cum = np.exp(logPf_cum_adj)

    # Equivalent beta via inverse survival (standard normal)
    normal = ot.Normal()
    beta_cum = np.vectorize(lambda p: normal.computeInverseSurvivalFunction(float(p))[0])(Pf_cum)

    return beta_cum, logPf_cum, Pf_cum
