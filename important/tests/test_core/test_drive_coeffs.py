"""
Tests for the scalar drive-coefficient functions.

These are the functions handed to ``qt.QobjEvo`` as time-dependent
coefficients, so a sign or cutoff error here silently changes every
simulation without raising anything.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------
# coeff / const
# --------------------------------------------------------------------------
def test_coeff_is_sine_drive(scar_functions):
    A, omega = 0.1, 0.6366896896896898

    for t in (0.0, 0.5, 1.0, 7.25, 100.0):
        assert scar_functions.coeff(t, A=A, omega=omega) == pytest.approx(
            A * np.sin(omega * t)
        )


def test_coeff_zero_at_origin_and_period(scar_functions):
    omega = 2.0
    assert scar_functions.coeff(0.0, A=1.0, omega=omega) == pytest.approx(0.0)
    assert scar_functions.coeff(np.pi / omega, A=1.0, omega=omega) == pytest.approx(
        0.0, abs=1e-12
    )


def test_coeff_amplitude_is_bounded(scar_functions):
    A = 0.1
    t = np.linspace(0, 200, 2001)
    values = np.array([scar_functions.coeff(x, A=A, omega=0.7) for x in t])
    assert np.max(np.abs(values)) <= A + 1e-12


def test_const_is_linear_ramp(scar_functions):
    for t in (0.0, 1.0, 3.5, 42.0):
        assert scar_functions.const(t, A=2.0) == pytest.approx(2.0 * t)


# --------------------------------------------------------------------------
# timed_drive / timed_const  (drive switched off after ``limit``)
# --------------------------------------------------------------------------
def test_timed_drive_matches_coeff_before_limit(scar_functions):
    A, omega, limit = 2.0, np.pi, 10.0

    for t in (0.0, 0.5, 4.0, 9.999):
        assert scar_functions.timed_drive(t, A=A, omega=omega, limit=limit) == (
            pytest.approx(scar_functions.coeff(t, A=A, omega=omega))
        )


def test_timed_drive_is_off_after_limit(scar_functions):
    for t in (10.0, 10.001, 50.0):
        assert scar_functions.timed_drive(t, A=2.0, omega=np.pi, limit=10.0) == 0.0


def test_timed_drive_switches_exactly_at_limit(scar_functions):
    """The cutoff is ``t < limit``, so t == limit is already off."""
    limit = 1.0
    assert scar_functions.timed_drive(limit, A=2.0, omega=1.0, limit=limit) == 0.0
    assert scar_functions.timed_drive(
        np.nextafter(limit, 0.0), A=2.0, omega=1.0, limit=limit
    ) != 0.0


def test_timed_const_matches_const_before_limit(scar_functions):
    assert scar_functions.timed_const(0.5, A=2.0, limit=1.0) == pytest.approx(1.0)
    assert scar_functions.timed_const(0.5, A=2.0, limit=1.0) == pytest.approx(
        scar_functions.const(0.5, A=2.0)
    )


def test_timed_const_is_off_after_limit(scar_functions):
    assert scar_functions.timed_const(1.0, A=2.0, limit=1.0) == 0.0
    assert scar_functions.timed_const(1.5, A=2.0, limit=1.0) == 0.0


# --------------------------------------------------------------------------
# make_coeff  (per-site drive frequency, used for frequency disorder)
# --------------------------------------------------------------------------
def test_make_coeff_reads_its_own_site_frequency(scar_functions):
    """
    ``make_coeff(r)`` must close over ``r`` and read ``args["wd{r}"]``.

    A late-binding bug here would make every site use the last site's
    frequency, which is exactly the failure mode that would make frequency
    disorder look like it has no effect.
    """
    args = {"A": 0.1, "wd0": 1.0, "wd1": 2.0, "wd2": 3.0}
    funcs = [scar_functions.make_coeff(r) for r in range(3)]

    t = 0.37
    for r, f in enumerate(funcs):
        assert f(t, args) == pytest.approx(0.1 * np.sin(args[f"wd{r}"] * t))


def test_make_coeff_agrees_with_coeff(scar_functions):
    """With one site, ``make_coeff`` must reproduce plain ``coeff``."""
    args = {"A": 0.1, "wd0": 0.6366896896896898}
    f = scar_functions.make_coeff(0)

    for t in (0.0, 1.0, 13.7):
        assert f(t, args) == pytest.approx(
            scar_functions.coeff(t, A=args["A"], omega=args["wd0"])
        )
