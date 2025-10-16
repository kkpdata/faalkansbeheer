"""Deze module bevat de calibratiefuncties voor de mechanismen piping en macrostabiliteit afgeleid in het WBI2017 project."""

import math
from dataclasses import dataclass

from scipy.stats import norm


def calc_Beta_uplift(F_u: float, Bnorm: float) -> float:
    r"""Berekening van de benaderde betrouwbaarheidsindex voor het mechanisme opbarsten (Uplift) gebaseerd op de WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::
        \beta_u = \frac{ln(F_{u}/0.48) + 0.27 \cdot \beta_{norm}}{0.46}


    Args:
        F_u (float): veiligheidsfactor voor opbarsten
        Bnorm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: benaderde betrouwbaarheidsindex voor het mechanisme opbarsten.
    """
    return (math.log(F_u / 0.48) + (0.27 * Bnorm)) / 0.46


def calc_Beta_heave(F_h: float, Bnorm: float) -> float:
    r"""Berekening van de benaderde betrouwbaarheidsindex voor het mechanisme Heave gebaseerd op de WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::

        \beta_h = \frac{ln(F_{h}/0.37) + 0.30 \cdot \beta_{norm}}{0.48}

    Args:
        F_h (float): veiligheidsfactor heave
        Bnorm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: benaderde betrouwbaarheidsindex voor het mechanisme heave.
    """
    return (math.log(F_h / 0.37) + (0.30 * Bnorm)) / 0.48


def calc_Beta_piping(F_p: float, Bnorm: float) -> float:
    r"""Berekening van de benaderde betrouwbaarheidsindex voor het mechanisme terugschreidende erosie gebaseerd op de WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::

        \beta_p = \frac{ln(F_{p}/1.04) + 0.43 \cdot \beta_{norm}}{0.37}


    Args:
        F_p (float): veiligheidsfactor piping
        Bnorm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: benaderde betrouwbaarheidsindex voor het mechanisme terugschreidende erosie.
    """
    return (math.log(F_p / 1.04) + (0.43 * Bnorm)) / 0.37


def calc_SF_uplift(B_cross: float, B_norm: float) -> float:
    r"""Berekening van de vereiste veiligheidsfactor voor het mechanisme opbarsten gebaseerd de op WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::

        \gamma_u = 0.48 \cdot e^{0.46 \cdot - \beta_{cross} - 0.27 \cdot - \beta_{norm}}


    Args:
        B_cross (float): vereiste betrouwbaarheidsindex voor opbarsten doorsnede eis (positieve value)
        Bnorm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: safety factor for uplift failure mechanism
    """
    return 0.48 * math.exp(0.46 * B_cross - 0.27 * B_norm)


def calc_SF_heave(B_cross: float, B_norm: float) -> float:
    r"""Berekening van de vereiste veiligheidsfactor voor het mechanisme heave gebaseerd op de WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::

        \gamma_h = 0.37 \cdot e^{0.48 \cdot - \beta_{cross} - 0.30 \cdot - \beta_{norm}}


    Args:
        B_cross (float): vereiste betrouwbaarheidsindex voor opbarsten doorsnede eis (positieve value)
        B_norm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: vereiste veiligheidsfactor voor het mechanisme heave
    """
    return 0.37 * math.exp(0.48 * B_cross - 0.30 * B_norm)


def calc_SF_piping(B_cross: float, B_norm: float) -> float:
    r"""Berekening van de vereiste veiligheidsfactor voor het mechanisme terugeschreidende erosie gebaseerd op de WBI2017 kalibratie.

    bron :cite:t:`calibration_piping_2016`

    math::

        \gamma_p = 1.04 \cdot e^{0.37 \cdot - \beta_{cross} - 0.43 \cdot - \beta_{norm}}


    Args:
        B_cross (float): vereiste betrouwbaarheidsindex voor opbarsten doorsnede eis (positieve value)
        B_norm (float): vereiste betrouwbaarheidsindex van het dijktraject (positieve waarde)

    Returns:
        float: vereiste veiligheidsfactor voor het mechanisme terugschreidende erosie
    """
    return 1.04 * math.exp(0.37 * B_cross - 0.43 * B_norm)


@dataclass
class ReliabilityDikeTrajectory:
    """Class om vereiste doorsnede eisen van een dijktraject te bepalen.

    Attributes:
        T (float): Terugkeertijd van de norm van het dijktraject [jaar]
        w (float): faalkansruimte [-]
        L (float): lengte van het dijktraject [m]
        a (float):  [1/m]
        b (float): parameter for the number of independent failure mechanisms per unit length [m]

    """

    T: float
    w: float
    L: float
    a: float
    b: float

    @property
    def Pnorm(self) -> float:
        return 1.0 / self.T

    @property
    def P_failure_mechanism(self) -> float:
        return self.w / self.T

    @property
    def B_norm(self) -> float:
        return float(-1.0 * norm.ppf(self.Pnorm))

    @property
    def B_failure_mechanism(self) -> float:
        return float(-1.0 * float(norm.ppf(self.P_failure_mechanism)))

    @property
    def N_dsn(self) -> float:
        return max(1.0, 1.0 + (self.a * self.L) / self.b)

    @property
    def P_cross(self) -> float:
        return self.P_failure_mechanism / self.N_dsn

    @property
    def B_cross(self) -> float:
        return float(-1.0 * norm.ppf(self.P_cross))
