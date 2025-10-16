"""Testfile for calibration functions in wbi.py"""

import pytest

from faalkansbeheer.helper_functions import calibration_functions_wbi as wbi

# beschrijving van de testcase
# Piping + Heave + Uplift
# T = 30000 jaar
# L = 30972.1 m
# w = 0.24
# a = 0.4
# b = 300 m
# N_traject = 42.296
# Pf_norm = 3.33333333e-05
# B_norm = 3.9878789366069176
# Pf_dsn = 1.891425851378061e-07
# B_dsn = 5.079572486324747
# SF_heave = 1.2809203
# SF_uplift = 1.6919925
# SF_erosion = 1.2261444

# Macrostabiliteit
# T = 30000 jaar
# L = 30972.1 m
# w = 0.04
# a = 0.033
# b = 50 m
# N_traject = 21.64807
# Pf_norm = 3.33333333e-05
# B_norm = 3.9878789366069176
# Pf_dsn = 6.218445470E-08
# B_dsn = 5.2887090184190
# SF_macro = 1.203306353


def test_ReliabilityDikeTrajectory():
    """Test de klasse ReliabilityDikeTrajectory."""
    traject_stph = wbi.ReliabilityDikeTrajectory(
        T=30000, w=0.24, L=30972.1, a=0.4, b=300
    )
    assert traject_stph.Pnorm == 1.0 / 30000
    assert traject_stph.B_norm == 3.9878789366069176
    assert traject_stph.P_failure_mechanism == 0.24 / 30000
    assert traject_stph.B_failure_mechanism == 4.314451021808664
    assert traject_stph.N_dsn == 42.29613333333334
    assert traject_stph.P_cross == 1.891425851378061e-07
    assert traject_stph.B_cross == 5.079572486324747


def test_calc_Beta_uplift():
    """Test de functie calc_Beta_Uplift."""
    F_u = 1.6919925
    B_norm = 3.9878789366069176
    B_uplift = wbi.calc_Beta_uplift(F_u, B_norm)
    assert B_uplift == pytest.approx(5.079572486324747, rel=1e-5)


def test_calc_Beta_heave():
    """Test de functie calc_Beta_Heave."""
    F_h = 1.2809203
    B_norm = 3.9878789366069176
    B_heave = wbi.calc_Beta_heave(F_h, B_norm)
    assert B_heave == pytest.approx(5.079572486324747, rel=1e-5)


def test_calc_Beta_piping():
    """Test de functie calc_Beta_Piping."""
    F_p = 1.2261444
    B_norm = 3.9878789366069176
    B_piping = wbi.calc_Beta_piping(F_p, B_norm)
    assert B_piping == pytest.approx(5.079572486324747, rel=1e-5)


def test_calc_SF_uplift():
    """Test de functie calc_SF_Uplift."""
    B_cross = 5.079572486324747
    B_norm = 3.9878789366069176
    SF_uplift = wbi.calc_SF_uplift(B_cross, B_norm)
    assert SF_uplift == pytest.approx(1.6919925, rel=1e-5)


def test_calc_SF_heave():
    """Test de functie calc_SF_Heave."""
    B_cross = 5.079572486324747
    B_norm = 3.9878789366069176
    SF_heave = wbi.calc_SF_heave(B_cross, B_norm)
    assert SF_heave == pytest.approx(1.2809203, rel=1e-5)


def test_calc_SF_piping():
    """Test de functie calc_SF_Piping."""
    B_cross = 5.079572486324747
    B_norm = 3.9878789366069176
    SF_piping = wbi.calc_SF_piping(B_cross, B_norm)
    assert SF_piping == pytest.approx(1.2261444, rel=1e-5)


def test_ReliabilityDikeTrajectory_macro():
    """Test de klasse ReliabilityDikeTrajectory voor macrostabiliteit."""
    traject_macro = wbi.ReliabilityDikeTrajectory(
        T=30000, w=0.04, L=30972.1, a=0.033, b=50
    )
    assert traject_macro.Pnorm == 1.0 / 30000
    assert traject_macro.B_norm == 3.9878789366069176
    assert traject_macro.P_failure_mechanism == 0.04 / 30000
    assert traject_macro.B_failure_mechanism == 4.694954045276703
    assert traject_macro.N_dsn == 21.441586
    assert traject_macro.P_cross == pytest.approx(6.218445470e-08, rel=1e-3)
    assert traject_macro.B_cross == pytest.approx(5.2887090184190, rel=1e-3)
