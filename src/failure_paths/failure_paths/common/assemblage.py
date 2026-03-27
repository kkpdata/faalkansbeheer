r"""Python module met functies voor het assemblageprotocol."""

import numpy as np
from decimal import Decimal, getcontext
from typing import cast


# noinspection PyPep8Naming
def bepaal_N_vak(L: float, a: float, dL: float) -> float:
    """Bepaalt de lengte-effect-factor N met een minimum van 1,0. Conform de Rode draad #10 assembleren (October 2024).

    :param L: Lengte van het element.
    :param a: Mechanismegevoelige fractie
    :param dL: De equivalente onafhankelijke lengte

    :raises ValueError: Parameter a moet groter zijn dan 0.
    :raises ValueError: De lengte L en dL moeten groter zijn dan 0.

    :return N_vak: Lengte-effect voor het vak
    """
    if a < 0:
        raise ValueError("a moet groter zijn dan 0.")

    if L < 0 or dL < 0:
        raise ValueError("De lengte L en dL moeten groter zijn dan 0.")

    N_vak = max(1.00, (a * L) / dL)
    return N_vak


def combine_series(list_pf: list[float]) -> tuple[float, float]:
    """Combineert de faalkansen uit een lijst naar:

     - een elementaire ondergrens op basis van volledige afhankelijkheid. De ondergrens wordt bepaald door max(pf[i]) van alle doorsneden.
     - een elementaire bovengrens op basis van volledige onafhankelijkheid. De bovengrens wordt bepaald door 1 - prod(1-pf[i]) van alle
       doorsneden.

    :param list_pf: Lijst met faalkansen van de elementen.
    :return bovengrens en ondergrens
    """
    # If empty
    if len(list_pf) == 0:
        return 0.0, 0.0

    # Berekenen van ondergrens
    ondergrens = max(list_pf)

    # We have to use Decimal for bovengrens
    getcontext().prec = 30
    # Because with small numbers (e-18 and smaller) it turns out that 1 - e-18 is rounded to one. Therefore, we have to
    # use Decimal with a lowered precision (we use up to e-30). We now first convert the necessary values to Decimal:
    one = Decimal(1)
    list_pf_dec = [Decimal(str(pf)) for pf in list_pf]

    # Berekenen van bovengrens
    list_pf_inv = [one - pf for pf in list_pf_dec]
    list_pf_inv = np.array(list_pf_inv, dtype=object)
    list_pf_inv = np.prod(list_pf_inv)
    list_pf_inv = cast(Decimal, list_pf_inv)
    bovengrens = float(one - list_pf_inv)

    return bovengrens, ondergrens
