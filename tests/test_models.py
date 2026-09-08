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
        "s0": np.array(500.0),
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
