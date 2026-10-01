import numpy as np
import pytest

from bayesbuilding.models import heating_cp_occ_rad


def test_heating_cp_occ_rad_floors_heat_and_computes_sigma():
    x = np.array(
        [
            [5.0, 0.0, 0],  # cold, no solar gain -> heating demand > 0
            [20.0, 500.0, 1],  # warm + large solar gain -> heat floored at 0
        ]
    )
    variables_dict = {
        "base": np.array([100.0, 200.0]),
        "g": np.array([10.0, 10.0]),
        "tau": np.array([18.0, 18.0]),
        "fs": np.array([1.0, 1.0]),
        "s0": np.array([500.0, 500.0]),
        "s1": np.array(50.0),
    }

    mu, extras = heating_cp_occ_rad(x, variables_dict)
    mu_val = mu.eval()
    sigma = extras["sigma"].eval()

    expected_heat_0 = 10.0 * (18.0 - 5.0)
    assert mu_val[0] == pytest.approx(100.0 + expected_heat_0)
    # Solar gain (500) far exceeds any heating demand -> heat floored at 0, not
    # negative, so it doesn't eat into the baseline.
    assert mu_val[1] == pytest.approx(200.0)

    # The model computes its own heteroscedastic sigma =
    # sqrt(s0**2 + (s1 * w)**2), where w = sigmoid((tau - t_ext) / 1.5) ramps
    # from 0 to 1 as it gets colder than the changepoint -- i.e. extra noise
    # (on top of the s0 floor) kicks in smoothly once heating is active.
    s0 = 500.0
    s1 = 50.0
    w0 = 1.0 / (1.0 + np.exp(-(18.0 - 5.0) / 1.5))
    w1 = 1.0 / (1.0 + np.exp(-(18.0 - 20.0) / 1.5))
    assert sigma[0] == pytest.approx(np.sqrt(s0**2 + (s1 * w0) ** 2))
    assert sigma[1] == pytest.approx(np.sqrt(s0**2 + (s1 * w1) ** 2))
    # Colder day -> more heating -> higher w -> more noise added to the floor.
    assert sigma[0] > sigma[1]


def test_heating_dt_occ_rad_multiroom_sums_over_rooms():
    """Direct call (no PyMC involved) checking the room axis is summed, not
    left dangling, and that occupation correctly selects a (n_obs, n_room)
    slice of a (n_occ, n_room) parameter via fancy indexing on the first axis
    only -- the failure mode this guards against is accidentally pairwise-
    indexing occupation against room (like season_cp_heating_es_rad_base_by_
    period's base[is_heating, period]) instead of broadcasting one against
    the other."""
    from bayesbuilding.models import heating_dt_occ_rad_multiroom

    n_obs, n_room = 5, 3
    x = np.zeros((n_obs, n_room + 3))
    x[:, :n_room] = np.arange(n_obs * n_room).reshape(n_obs, n_room)
    x[:, n_room] = 1.0  # rad
    x[:, n_room + 1] = 0.0  # elec_consumption
    x[:, n_room + 2] = [0, 1, 0, 1, 0]  # occupation

    g = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    fs = np.zeros((2, n_room))
    variables_dict = {
        "g": g,
        "fs": fs,
        "alpha": np.array([0.0, 0.0]),
        "s0": np.array([1.0, 1.0]),
    }

    mu, extras = heating_dt_occ_rad_multiroom(x, variables_dict)

    dt = x[:, :n_room]
    occ = x[:, -1].astype(int)
    expected = (g[occ] * dt).sum(axis=1)

    assert mu.shape == (n_obs,)
    np.testing.assert_allclose(mu, expected)



def test_heating_dt_occ_rad_lag_reheat_surcharge_sign():
    """Direct call (no PyMC) locking down the sign convention for h, worked
    out from the underlying physics (see bayes.py docstring / plan): with a
    thermal-mass recharge cost C/dt_step = 50 folded into g_fit=150, h=-50 on
    top of a steady-state g=100, a stable day (dt == dt_lag == 10) must
    reproduce the plain g*dt = 1000, while a reheat day coming out of a low
    dt_lag=2 up to dt=10 must show the extra +400 surcharge (1400 total) --
    catches an accidental sign flip on h that unit shape/smoke tests alone
    wouldn't."""
    from bayesbuilding.models import heating_dt_occ_rad_lag

    g_fit, h_fit = 150.0, -50.0
    x = np.array(
        [
            [10.0, 0.0, 0.0, 1, 10.0],  # stable day: dt == dt_lag
            [10.0, 0.0, 0.0, 1, 2.0],  # reheat day: dt_lag much lower than dt
        ]
    )
    variables_dict = {
        "g": np.array([g_fit, g_fit]),
        "fs": np.array([0.0, 0.0]),
        "alpha": np.array([0.0, 0.0]),
        "h": np.array([h_fit, h_fit]),
        "s0": np.array([1.0, 1.0]),
    }

    mu, extras = heating_dt_occ_rad_lag(x, variables_dict)

    np.testing.assert_allclose(mu, [1000.0, 1400.0])



def test_heating_dt_occ_rad_DTdt_lags_capacity_construction():
    """Direct call (no PyMC) locking down the c0 * rho**k geometric-decay
    construction and the lag-column order: with c0=100 and rho=0.5, the
    per-lag coefficients must be [100, 50, 25, 12.5] applied to DT_dt at
    [lag0 (today), lag1, lag2, lag3] in that order -- catches an accidental
    transpose of the lag columns or an off-by-one in the decay exponent that
    a shape/smoke test alone wouldn't. Also locks down that every coefficient
    stays positive (guaranteed by construction for c0 > 0, 0 < rho < 1),
    unlike the earlier c0 + cumsum(d) version, whose unconstrained increments
    let a higher lag's coefficient go negative -- an unphysical sign flip for
    a fading capacitive effect."""
    from bayesbuilding.models import heating_dt_occ_rad_DTdt_lags

    c0_fit = 100.0
    rho_fit = 0.5
    dt_dt_lags = np.array([1.0, 2.0, 3.0, 4.0])  # lag0, lag1, lag2, lag3

    x = np.array([[10.0, 0.0, 0.0, 1, *dt_dt_lags]])
    variables_dict = {
        "g": np.array([0.0, 0.0]),
        "fs": np.array([0.0, 0.0]),
        "alpha": np.array([0.0, 0.0]),
        "c0": c0_fit,
        "rho": rho_fit,
        "s0": np.array([1.0, 1.0]),
    }

    mu, extras = heating_dt_occ_rad_DTdt_lags(x, variables_dict)

    c_expected = np.array([100.0, 50.0, 25.0, 12.5])
    assert (c_expected > 0).all()
    expected = dt_dt_lags @ c_expected

    np.testing.assert_allclose(mu, [expected])



def test_heating_dt_occ_rad_DTdt_wall_Ci_adds_air_capacity_term():
    """Direct call (no PyMC) locking down the sign/construction of the new
    Ci*DTint_intraday term: with every other coefficient zeroed out (g, fs,
    alpha, C all 0, beta arbitrary since C=0 makes it irrelevant), mu must
    equal exactly Ci * DTint_intraday -- catches an accidental sign flip or a
    swapped column index for the new 7th input column."""
    from bayesbuilding.models import heating_dt_occ_rad_DTdt_wall_Ci

    Ci_fit = 8000.0
    DTint_intraday = np.array([1.5, -0.5])

    x = np.array(
        [
            [10.0, 0.0, 0.0, 1, 0.0, 0.0, DTint_intraday[0]],
            [10.0, 0.0, 0.0, 0, 0.0, 0.0, DTint_intraday[1]],
        ]
    )
    variables_dict = {
        "g": np.array([0.0, 0.0]),
        "fs": np.array([0.0, 0.0]),
        "alpha": np.array([0.0, 0.0]),
        "C": 0.0,
        "beta": 0.5,
        "Ci": Ci_fit,
        "s0": np.array([1.0, 1.0]),
    }

    mu, extras = heating_dt_occ_rad_DTdt_wall_Ci(x, variables_dict)

    np.testing.assert_allclose(mu, Ci_fit * DTint_intraday)
