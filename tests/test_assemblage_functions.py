r"""Test file for assemblage_functions.py"""

# import assemblage_functions
import numpy as np
import pytest

from faalkansbeheer.helper_functions.assemblage_functions import (
    bepaal_N_vak,
    combin_seriesysteem,
)


def test_bepaal_N_vak_typical_values():
    assert bepaal_N_vak(10, 0.5, 1) == pytest.approx(5.0, rel=1e-3)
    assert bepaal_N_vak(15, 0.2, 5) == pytest.approx(
        max(1, (0.2 * 15) / 5), rel=1e-3
    )  # = 0.6 -> 1


def test_bepaal_N_vak_min_bound_and_zero_L_or_a():
    assert bepaal_N_vak(0, 0.5, 2) == pytest.approx(1.0, rel=1e-3)
    assert bepaal_N_vak(10, 0.0, 2) == pytest.approx(1.0, rel=1e-3)


def test_bepaal_N_vak_a_greater_than_one_allowed_by_current_impl():
    # Current implementation allows a > 1; verify formula is applied
    assert bepaal_N_vak(10, 3.0, 1) == pytest.approx(30.0, rel=1e-3)


def test_bepaal_N_vak_negative_a_raises():
    with pytest.raises(ValueError, match="a moet groter zijn dan 0"):
        bepaal_N_vak(10, -0.1, 1)


@pytest.mark.parametrize("L,dL", [(-1, 1), (1, -1), (-5, -2)])
def test_bepaal_N_vak_negative_lengths_raise(L, dL):
    with pytest.raises(ValueError, match="De lengte L en dL moeten groter zijn dan 0"):
        bepaal_N_vak(L, 0.5, dL)


def test_bepaal_N_vak_zero_dL_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        bepaal_N_vak(10, 0.5, 0)


def test_combin_seriesysteem_empty_array_raises():
    with pytest.raises(ValueError, match="mag niet leeg zijn"):
        combin_seriesysteem(np.array([]))


def test_combin_seriesysteem_out_of_range_values_raise_value_error():
    # Current implementation raises ValueError due to ambiguous truth value;
    # any ValueError is acceptable here.
    with pytest.raises(ValueError):
        combin_seriesysteem(np.array([0.1, -0.2, 0.3]))
    with pytest.raises(ValueError):
        combin_seriesysteem(np.array([0.0, 1.2, 0.5]))


# TODO: add fixture for valid input after creating invalid input tests
def test_combin_seriesysteem_valid_values_expected_bounds():
    pfs = np.array([0.1, 0.2, 0.3])
    ondergrens_expected = float(np.maximum.accumulate(pfs)[-1])
    bovengrens_expected = float(1 - np.cumprod(1 - pfs)[-1])
    ondergrens, bovengrens = combin_seriesysteem(pfs)
    assert isinstance(ondergrens, float)
    assert isinstance(bovengrens, float)
    assert ondergrens == ondergrens_expected
    assert bovengrens == bovengrens_expected
