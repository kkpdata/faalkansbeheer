import scipy.stats as sct

TRAJECT_PROPERTIES = {
    "16-1": {"signaleringswaarde": 100_000, "ondergrens": 30_000, "lengte": 15_059.41},
    "16-2": {"signaleringswaarde": 30_000, "ondergrens": 10_000, "lengte": 30972.09},
}


class TrajectNormering:
    """Gathers the traject id and calculates the traject normering from the HRD-files."""

    def __init__(
        self,
        traject_id: str,  # Bijvoorbeeld '16-1'
        norm_is_ondergrens: bool = True,
        bovenrivierengebied: bool = True,
    ):
        # Input
        self.bovenrivierengebied: bool = bovenrivierengebied

        # Parameters
        self.traject_id: str = traject_id
        self.signaleringswaarde: int = TRAJECT_PROPERTIES[traject_id]["signaleringswaarde"]
        self.ondergrens: int = TRAJECT_PROPERTIES[traject_id]["ondergrens"]
        self.w: float = 0.24
        self.traject_lengte: float = TRAJECT_PROPERTIES[traject_id]["lengte"]
        self.faalkanseis_signaleringswaarde = 1.0 / self.signaleringswaarde
        self.faalkanseis_ondergrens = 1.0 / self.ondergrens
        self.faalkanseis_norm = self.faalkanseis_ondergrens
        if not norm_is_ondergrens:
            self.faalkanseis_norm = self.signaleringswaarde
        self.beta_norm = sct.norm.ppf(self.faalkanseis_norm)
        self.n_dsn = 1 + (0.9 * self.traject_lengte) / 300.0
        if not self.bovenrivierengebied:
            self.n_dsn = 1 + (0.4 * self.traject_lengte) / 300.0
        # TODO Nu Must Klein: Eigenlijk hoofdletter N_dsn.
        # Maar ipv afkorting naam gebruiken?
        self.faalkanseis_sign_dsn = (self.w * self.faalkanseis_signaleringswaarde) / self.n_dsn
        self.beta_sign_dsn = sct.norm.ppf(self.faalkanseis_sign_dsn)
        self.faalkanseis_ond_dsn = (self.w * self.faalkanseis_ondergrens) / self.n_dsn
        self.beta_ond_dsn = sct.norm.ppf(self.faalkanseis_ond_dsn)
        self.beta_categorie_grenzen = {
            "I": [-1 * sct.norm.ppf(self.faalkanseis_sign_dsn / 30), 50],
            "II": [
                -1 * sct.norm.ppf(self.faalkanseis_sign_dsn),
                -1 * sct.norm.ppf(self.faalkanseis_sign_dsn / 30),
            ],
            "III": [
                -1 * sct.norm.ppf(self.faalkanseis_ond_dsn),
                -1 * sct.norm.ppf(self.faalkanseis_sign_dsn),
            ],
            "IV": [
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens),
                -1 * sct.norm.ppf(self.faalkanseis_ond_dsn),
            ],
            "V": [
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens * 30),
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens),
            ],
            "VI": [
                -50,
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens * 30),
            ],
        }
        self.riskeer_categorie_grenzen = {
            "+III": [-1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 1000), 20],
            "+II": [
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 100),
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 1000),
            ],
            "+I": [
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 10),
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 100),
            ],
            "0": [
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde),
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde / 10),
            ],
            "-I": [
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens),
                -1 * sct.norm.ppf(self.faalkanseis_signaleringswaarde),
            ],
            "-II": [
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens * 10),
                -1 * sct.norm.ppf(self.faalkanseis_ondergrens),
            ],
            "-III": [2, -1 * sct.norm.ppf(self.faalkanseis_ondergrens * 10)],
        }
