r"""Python module om de elementaire bovengrenzen en ondergrenzen van een seriesysteem te berekenen."""

import numpy as np


def bepaal_N_vak(L: float, a: float, dL: float) -> float:
    r"""Bepaalt de opschaalfactor per vak.

    Parameters
    ----------
    L : float
        De lengte van het vak [m]
    a : float
        deel van het vak waarvoor de opschaalfactor wordt berekend (0-1).
    dL : float
        equivalente onafhankelijke mechanismelengte [m].

    Returns
    -------
    float
        De opschaalfactor van het vak

    Raises
    ------
    ValueError
        Als a niet tussen 0 en 1 ligt, of als L en dL niet positief zijn.


    """

    if a < 0:
        raise ValueError("a moet groter zijn dan 0.")

    if L < 0 or dL < 0:
        raise ValueError("De lengte L en dL moeten groter zijn dan 0.")

    N_vak = max(1, (a * L) / dL)
    return N_vak


def combin_seriesysteem(pfs: np.ndarray) -> tuple[float, float]:
    r"""Bereken de elementaire bovengrenzen en ondergrenzen van een seriesysteem.

    Parameters
    ----------
    pf : np.ndarray
        Een array van faalkansen voor de componenten in het seriesysteem.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Een tuple met twee arrays: de eerste bevat de elementaire ondergrenzen,
        de tweede bevat de elementaire bovengrenzen.

    Raises
    ------
    ValueError
        Als de input array leeg is of negatieve waarden bevat.


    """

    if pfs.size == 0:
        raise ValueError("De input array mag niet leeg zijn.")

    if np.any(pfs < 0) or np.any(pfs > 1):
        raise ValueError("Faalkansen moeten tussen 0 en 1 liggen.")

    # bereken ondergrens seriersysteem
    ondergrens = pfs.max(axis=0)
    # bereken bovengrens seriersysteem
    bovengrens = 1 - np.prod(1 - pfs, axis=0)
    return ondergrens, bovengrens
