import pymc as pm
import pytensor.tensor as pt

# STC air temperature [°C]
STC_AIR_TEMP = 25

# STC irradiance [W/m2]
STC_IRRADIANCE = 1000

# NOCT air temperature [°C]
NOCT_AIR_TEMP = 20

# NOCT irradiance [W/m2]
NOCT_IRRADIANCE = 800


def occ_cp(x, variables_dict: dict):
    """
    Model a system assuming it has two distinct fairly constant behaviour
    during specified period. For example week-days and weekends, or holidays.
    :param x: The features. x[:, 0] must be a columns of boolean values, or integer
    indicating the period (from 0, to n period)
    :param variables_dict: variable dictionary with a unique distribution called
    set_point of shape n period
    :return:
    """
    wd_we = x[:, 0].astype(int)
    set_point = variables_dict["set_point"]
    return set_point[wd_we], {"sigma": variables_dict["sigma"]}


def season_cp_occ_cp_heating_cooling_es(x, variables_dict: dict):
    """
    Season Occupation Change point Heating / Cooling Energy Signature
    source S. Rouchier (https://buildingenergygeeks.org/bayesianmv.html)
    Piecewise linear model to predict the overall building energy consumption.
    Assume 3 distinct periods Heating, Cooling, mid-season:
    n distinct functions for n occupation typology.

    if tay_heat > t_ext :
        E = g_{heat} * (tau_{heat} - t_{ext}) + base

    if tau_cool > t_ext :
        E = g_{cool} * (t_{ext} - tau_{cool}) + base

    else :
        E = base

    :param x: The features . x[:, 0] must be a columns of boolean values, or integer
    indicating the period (from 0, to n period), x[:, 1] must be external temperatures
    :param variables_dict: mandatory model variables are : "base", "g_h", "g_c",
    "tau_h", "tau_c"
    :return: Energy consumption
    """
    occupation = x[:, 0].astype(int)
    t_ext = x[:, 1]

    base = variables_dict["base"]
    g_h = variables_dict["g_h"]
    g_c = variables_dict["g_c"]
    tau_h = variables_dict["tau_h"]
    tau_c = variables_dict["tau_c"]

    baseline = base[occupation]
    heat = g_h[occupation] * pm.math.maximum(tau_h[occupation] - t_ext, 0)
    cool = g_c[occupation] * pm.math.maximum(t_ext - tau_c[occupation], 0)
    return baseline + heat + cool, {"sigma": variables_dict["sigma"]}


def season_cp_heating_es(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature

    During winter the overall building energy consumption is modeled as a linear
    function : Text * G + baseline. Where G is the overall heat loss [kWh/°C]
    and baseline is the "process" energy consumption.
    During summer, only baseline remains.
    The changepoint tau is based on the exterior temperature

    :param x: single column 2D array. x[:, 0] is the external air temperature
    :param variable_dict: mandatory model variables are : "g", "tau", "base"
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    consumption = g * pm.math.maximum(tau - t_ext, 0)
    return consumption + baseline, {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def season_cp_heating_es_rad(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature

    While heating is active the overall building energy consumption is modeled
    as a linear function : g * max(tau - Text, 0) - fs * rad + baseline[0].
    Where g is the overall heat loss [kWh/°C], fs discounts solar gains, and
    baseline[0] is the "process" energy consumption.
    While heating is off (is_heating == 0, e.g. summer or an explicit
    heating-absence period), consumption falls back to the flat baseline[1],
    with no weather dependency.
    The changepoint tau is based on the exterior temperature.

    :param x: 3-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the solar radiation, x[:, 2] is the heating-on flag
        (1 heating active, 0 heating off)
    :param variable_dict: mandatory model variables are : "g", "fs", "tau",
        "base" ("base" has shape 2: base[0] while heating, base[1] while off)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    is_heating = x[:, 2].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    return pm.math.switch(
        is_heating,
        baseline[0] + g * pm.math.maximum(tau - t_ext, 0) - fs * rad,
        baseline[1],
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def season_cp_heating_es_setback(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature, gated by a setback/heating
    state flag (no solar-gain term -- see season_cp_heating_es_rad for that).

    While heating is active: E = base[0] + g * max(tau - Text, 0).
    While heating is off (is_heating == 0, e.g. summer or an explicit
    heating-absence period): E falls back to the flat baseline[1], with no
    weather dependency.
    The changepoint tau is based on the exterior temperature.

    :param x: 2-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the heating-on flag (1 heating active, 0 heating off)
    :param variable_dict: mandatory model variables are : "g", "tau", "base"
        ("base" has shape 2: base[0] while heating, base[1] while off)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    is_heating = x[:, 1].astype(int)
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    return pm.math.switch(
        is_heating,
        baseline[0] + g * pm.math.maximum(tau - t_ext, 0),
        baseline[1],
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def season_cp_heating_es_rad_g_by_period(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature, g varying by period.

    Same as season_cp_heating_es_rad, but the overall heat loss coefficient g
    is allowed to differ across n distinct calendar periods (e.g. distinct
    fitting/monitoring campaigns), while fs, tau and base stay shared scalars
    (base still varies by heating-state only, not by period). Diagnostic
    variant used to test whether cross-period differences in the fitted g are
    attributable specifically to g, vs to tau or base (see
    tau_by_period / base_by_period siblings).

    :param x: 4-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the solar radiation, x[:, 2] is the heating-on flag
        (1 heating active, 0 heating off), x[:, 3] is the period index
        (0 to n-1)
    :param variable_dict: mandatory model variables are : "g" (shape n_period),
        "fs", "tau" (shared scalars), "base" (shape 2: base[0] while heating,
        base[1] while off, shared across periods)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    is_heating = x[:, 2].astype(int)
    period = x[:, 3].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    return pm.math.switch(
        is_heating,
        baseline[0] + g[period] * pm.math.maximum(tau - t_ext, 0) - fs * rad,
        baseline[1],
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def season_cp_heating_es_rad_tau_by_period(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature, tau varying by period.

    Same as season_cp_heating_es_rad, but the changepoint tau is allowed to
    differ across n distinct calendar periods, while g, fs and base stay
    shared scalars (base still varies by heating-state only, not by period).
    Diagnostic variant, see season_cp_heating_es_rad_g_by_period.

    :param x: 4-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the solar radiation, x[:, 2] is the heating-on flag
        (1 heating active, 0 heating off), x[:, 3] is the period index
        (0 to n-1)
    :param variable_dict: mandatory model variables are : "g", "fs" (shared
        scalars), "tau" (shape n_period), "base" (shape 2: base[0] while
        heating, base[1] while off, shared across periods)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    is_heating = x[:, 2].astype(int)
    period = x[:, 3].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    return pm.math.switch(
        is_heating,
        baseline[0] + g * pm.math.maximum(tau[period] - t_ext, 0) - fs * rad,
        baseline[1],
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def season_cp_heating_es_rad_base_by_period(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature, base varying by period.

    Same as season_cp_heating_es_rad, but the baseline is indexed by both
    heating-state and period (shape 2 x n_period), while g, fs and tau stay
    shared scalars. Diagnostic variant, see
    season_cp_heating_es_rad_g_by_period.

    :param x: 4-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the solar radiation, x[:, 2] is the heating-on flag
        (1 heating active, 0 heating off), x[:, 3] is the period index
        (0 to n-1)
    :param variable_dict: mandatory model variables are : "g", "fs", "tau"
        (shared scalars), "base" (shape (2, n_period): base[0, p] while
        heating in period p, base[1, p] while off in period p)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    is_heating = x[:, 2].astype(int)
    period = x[:, 3].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    tau = variable_dict["tau"]
    base = variable_dict["base"]

    baseline = base[is_heating, period]
    return pm.math.switch(
        is_heating,
        baseline + g * pm.math.maximum(tau - t_ext, 0) - fs * rad,
        baseline,
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def season_cp_heating_es_rad_g_tau_by_period(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature, g AND tau varying by period.

    Same as season_cp_heating_es_rad, but both the heat loss coefficient g and
    the changepoint tau are allowed to differ across n distinct calendar
    periods, while fs and base stay shared scalars (base still varies by
    heating-state only, not by period). Diagnostic variant: tests, via LOO
    against tau_by_period (tau alone varying), whether letting g *also* vary
    per period adds real explanatory power once tau already does -- i.e.
    whether the cross-period difference is carried by tau alone or needs g's
    own period-specific value too. See season_cp_heating_es_rad_g_by_period /
    season_cp_heating_es_rad_tau_by_period for the single-parameter siblings.

    :param x: 4-column 2D array. x[:, 0] is the external air temperature,
        x[:, 1] is the solar radiation, x[:, 2] is the heating-on flag
        (1 heating active, 0 heating off), x[:, 3] is the period index
        (0 to n-1)
    :param variable_dict: mandatory model variables are : "g" (shape
        n_period), "tau" (shape n_period), "fs" (shared scalar), "base"
        (shape 2: base[0] while heating, base[1] while off, shared across
        periods)
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    is_heating = x[:, 2].astype(int)
    period = x[:, 3].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    return pm.math.switch(
        is_heating,
        baseline[0] + g[period] * pm.math.maximum(tau[period] - t_ext, 0) - fs * rad,
        baseline[1],
    ), {"sigma": variable_dict["sigma"], "lower": pt.constant(0.0)}


def heating_es_dju(x, variable_dict):
    """
    Linear heating energy signature driven by degree-days (DJU).

    The overall building energy consumption is modeled as a linear function
    of the heating degree-days (DJU). The consumption is given by:

        consumption = base + g * dju

    where:
      - g is the slope (energy per unit of DJU),
      - base is the constant baseline (non-weather-dependent) energy
        consumption.

    :param x: 2D array with a single feature column.
        x[:, 0] contains the heating degree-days (DJU).
    :param variable_dict: Dictionary containing the model parameters:
        - "g": slope of the energy signature [energy/DJU]
        - "base": baseline energy consumption [energy]
    :return: Overall building energy consumption.
    """
    dju = x[:, 0]
    g = variable_dict["g"]
    baseline = variable_dict["base"]

    consumption = g * dju
    return consumption + baseline, {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def heating_es_dju_rad(x, variable_dict):
    """
    Linear heating energy signature driven by degree-days (DJU).

    The overall building energy consumption is modeled as a linear function
    of the heating degree-days (DJU). The consumption is given by:

        consumption = base + g * dju

    where:
      - g is the slope (energy per unit of DJU),
      - base is the constant baseline (non-weather-dependent) energy
        consumption.

    :param x: 2D array with a single feature column.
        x[:, 0] contains the heating degree-days (DJU).
    :param variable_dict: Dictionary containing the model parameters:
        - "g": slope of the energy signature [energy/DJU]
        - "base": baseline energy consumption [energy]
    :return: Overall building energy consumption.
    """
    dju = x[:, 0]
    rad = x[:, 1]
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    baseline = variable_dict["base"]

    return baseline + g * dju - fs * rad, {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def heating_es_dju_rad_occ(x, variable_dict):
    """
    Linear heating energy signature driven by degree-days (DJU).

    The overall building energy consumption is modeled as a linear function
    of the heating degree-days (DJU). The consumption is given by:

        consumption = base + g * dju

    where:
      - g is the slope (energy per unit of DJU),
      - base is the constant baseline (non-weather-dependent) energy
        consumption.

    :param x: 2D array with a single feature column.
        x[:, 0] contains the heating degree-days (DJU).
    :param variable_dict: Dictionary containing the model parameters:
        - "g": slope of the energy signature [energy/DJU]
        - "base": baseline energy consumption [energy]
    :return: Overall building energy consumption.
    """
    dju = x[:, 0]
    rad = x[:, 1]
    occ = x[:, 2].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    baseline = variable_dict["base"]

    return baseline[occ] + g[occ] * dju - fs[occ] * rad, {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def heating_es_dju_rad_occ_setback(x, variable_dict):
    """
    Heating energy signature gated by occupancy: the full weather-driven
    signature applies while occupied, and a flat setback baseline applies while
    not -- unlike heating_es_dju_rad_occ, which fits a fully separate g/fs/base
    per occupancy state, here only the baseline switches; the slope (g) and
    solar gain discount (fs) are shared across states, and the unoccupied state
    has no dju/rad dependency at all (e.g. heating setback/off outside
    occupancy).

        if occ == 1: E = base[0] + g * dju - fs * rad
        if occ == 0: E = base[1]

    :param x: 2D array with 3 feature columns.
        x[:, 0] contains the heating degree-days (DJU).
        x[:, 1] contains the solar radiation.
        x[:, 2] contains the occupancy period (0 or 1).
    :param variable_dict: Dictionary containing the model parameters:
        - "g": slope of the energy signature [energy/DJU], shared across states
        - "fs": solar gain discount factor, shared across states
        - "base": shape-2 baseline, "base[0]" while occupied (added to the
          weather-driven terms), "base[1]" the flat unoccupied baseline
    :return: Overall building energy consumption.
    """
    dju = x[:, 0]
    rad = x[:, 1]
    occ = x[:, 2].astype(int)
    g = variable_dict["g"]
    fs = variable_dict["fs"]
    base = variable_dict["base"]

    return pm.math.switch(occ, base[0] + g * dju - fs * rad, base[1]), {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def season_cp_heating_es_dt(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature (dt driven)

    The overall building energy consumption is modeled as a linear function
    of dt (indoor/outdoor temperature difference, tin - text). Below the
    changepoint tau, free heat gains (occupancy, equipment, solar) cover the
    envelope losses and the heating term is 0. Above tau, the heating term
    grows linearly with the excess (dt - tau), scaled by the overall heat
    loss coefficient g [kWh/°C]. baseline is the "process" energy
    consumption, added regardless of dt.

    :param x: single column 2D array. x[:, 0] is dt (tin - text)
    :param variable_dict: mandatory model variables are : "g", "tau", "base"
    :return: overall building consumption
    """
    dt = x[:, 0]
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    consumption = g * pm.math.maximum(dt - tau, 0)
    return consumption + baseline, {
        "sigma": variable_dict["sigma"],
        "lower": pt.constant(0.0),
    }


def season_cp_occ_cp_es_dt(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature

    During winter the overall building energy consumption is modeled as a linear
    function : Text * G + baseline. Where G is the overall heat loss [kWh/°C]
    and baseline is the "process" energy consumption.
    During summer, only baseline remains.
    The changepoint tau is based on the exterior temperature
    n distinct function for n occupation typologie

    :param x: 2D array. x[:, 0] is the occupation index, x[:, 1] is the esternal
    air temperature
    :param variable_dict: mandatory model variables are : "g", "tau", "base". Each
    variables have a shape of diemension n.
    :return: energy consumption
    """
    occ = x[:, 0].astype(int)
    dt = x[:, 1]
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["baseline"]

    consumption = g[occ] * pm.math.maximum(dt - tau[occ], 0)
    # sigma is computed by state here (replaces the removed
    # sigma_change_point_idx wrapper mechanism): the prior declares "sigma"
    # with shape n_occupation_states, and this model indexes it itself.
    sigma = variable_dict["sigma"][occ]
    return consumption + baseline[occ], {"sigma": sigma, "lower": pt.constant(0.0)}


def season_cp_occ_cp_heating_es(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature

    During winter the overall building energy consumption is modeled as a linear
    function : Text * G + baseline. Where G is the overall heat loss [kWh/°C]
    and baseline is the "process" energy consumption.
    During summer, only baseline remains.
    The changepoint tau is based on the exterior temperature
    n distinct function for n occupation typologie

    :param x: 2D array. x[:, 0] is the occupation index, x[:, 1] is the esternal
    air temperature
    :param variable_dict: mandatory model variables are : "g", "tau", "base". Each
    variables have a shape of diemension n.
    :return: energy consumption
    """
    occ = x[:, 0].astype(int)
    t_ext = x[:, 1]
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["baseline"]

    consumption = g[occ] * pm.math.maximum(tau[occ] - t_ext, 0)
    # sigma is computed by state here (replaces the removed
    # sigma_change_point_idx wrapper mechanism): the prior declares "sigma"
    # with shape n_occupation_states, and this model indexes it itself.
    sigma = variable_dict["sigma"][occ]
    return consumption + baseline[occ], {"sigma": sigma, "lower": pt.constant(0.0)}


def we_cst_wd_radiation_lighting(x, variable_dict):
    """
    Artificial Lighting energy consumption model.
    Depending on the occupation, returns a baseline_we consumption (weekends, holidays),
    or an external radiation dependant term + baseline_wd

    :param variable_dict: mandatory model variables are :
    - base_we: baseline with no occupation
    - base_wd: baseline during weekdays
    - fs: coefficient to account for natural lighting based on solar radiations
    :param x: x[:, 0] is boolean series or 0-1 describing 2 occupation behaviour,
        x[:, 1] is solar radiation. For solar radiation, use Global horizontal,
        or custom projection.
    """

    fs = variable_dict["fs"]
    base_we = variable_dict["base_we"]
    base_wd = variable_dict["base_wd"]

    return pm.math.switch(x[:, 0], base_we, base_wd + fs * x[:, 1]), {
        "sigma": variable_dict["sigma"]
    }


def season_cp_occ_cp_rad_heating_cooling_es(x, variables_dict: dict):
    """
    Add radiation to Season Occupation Change point Heating / Cooling Energy Signature
    source S. Rouchier (https://buildingenergygeeks.org/bayesianmv.html)
    Piecewise linear model to predict the overall building energy consumption.
    Assume 3 distinct periods Heating, Cooling, mid-season:
    n distinct functions for n occupation typology.

    if tau_heat > t_ext :
        heat = g_{heat} * (tau_{heat} - t_{ext})

    if tau_cool < t_ext :
        cool = g_{cool} * (t_{ext} - tau_{cool})

    if tau_rad_h > rad:
        solar_heat = fs_{heat} * (taurad_{heat} - rad)

    if tau_rad_c < rad:
        solar_cool = - fs_{heat} * (taurad_{cool} - rad)

    E = base

    :param x: The features . x[:, 0] must be a columns of boolean values, or integer
    indicating the period (from 0, to n period), x[:, 1] must be external temperatures,
    x[:, 2] is a measure of solar radiation. For exemple projected radiation or
    Global Horizontal depending on the value and the meaning of fs
    :param variables_dict: mandatory model variables are : "base", "g_h", "g_c",
    "tau_h", "tau_c", "fs_h", "fs_c", "tau_rad_h", "tau_rad_c".
    :return: Energy consumption
    """
    occupation = x[:, 0].astype(int)
    t_ext = x[:, 1]
    rad = x[:, 2]

    base = variables_dict["base"]
    g_h = variables_dict["g_h"]
    g_c = variables_dict["g_c"]
    tau_h = variables_dict["tau_h"]
    tau_c = variables_dict["tau_c"]
    fs_h = variables_dict["fs_h"]
    fs_c = variables_dict["fs_c"]
    tau_rad_h = variables_dict["tau_rad_h"]
    tau_rad_c = variables_dict["tau_rad_c"]

    baseline = base[occupation]
    solar_heat = fs_h[occupation] * pm.math.maximum(tau_rad_h[occupation] - rad, 0)
    solar_cool = -fs_c[occupation] * pm.math.maximum(rad - tau_rad_c[occupation], 0)
    heat = g_h[occupation] * pm.math.maximum(tau_h[occupation] - t_ext, 0)
    cool = g_c[occupation] * pm.math.maximum(t_ext - tau_c[occupation], 0)
    return baseline + heat + cool + solar_cool + solar_heat, {
        "sigma": variables_dict["sigma"]
    }


def heating_cp_occ_rad(x, variables_dict: dict):
    """
    Heating changepoint (tau - t_ext) net of solar gains (fs * rad), floored at
    zero, plus a per-occupation baseline. Solar gains are subtracted inside the
    max() so a large summer radiation surplus cannot make the term negative and
    eat into the baseline.

    The likelihood scale is heteroscedastic and computed by the model itself:
    sigma = sqrt(s0**2 + (s1 * w)**2), a noise floor (s0) combined in
    quadrature with a term (s1) scaled by w, a sigmoid of (tau - t_ext) that
    ramps from 0 to 1 as it gets colder than the changepoint -- i.e. extra
    noise kicks in smoothly once heating is actually active, instead of a
    single sigma shared regardless of how much the building was heating that
    day. "sigma" is a reserved key in the returned extras dict:
    PymcWrapper.build_model uses it in place of variables_dict["sigma"]
    whenever a model_function provides it (see its docstring).

    Returns (mu, extras) with extras = {"sigma": ...}.
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    occupation = x[:, 2].astype(int)

    base = variables_dict["base"]
    g = variables_dict["g"]
    tau = variables_dict["tau"]
    fs = variables_dict["fs"]
    s0 = variables_dict["s0"]
    s1 = variables_dict["s1"]

    baseline = base[occupation]
    heat = pm.math.maximum(
        g[occupation] * (tau[occupation] - t_ext) - fs[occupation] * rad, 0
    )

    mu = baseline + heat

    w = pm.math.sigmoid((tau[occupation] - t_ext) / 1.5)

    sigma = pt.sqrt(s0[occupation] ** 2 + (s1 * w) ** 2)

    return baseline + heat, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad(x, variables_dict: dict):
    """
    Consumption driven directly by the indoor/outdoor delta-T (dt = tin - text,
    no changepoint/floor here since dt is already ~0 or negative outside the
    heating season), net of solar gains (fs * rad) and of a fraction (alpha) of
    metered electrical consumption (a proxy for internal/appliance heat gains
    offsetting the heating load), per-occupation.

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    Returns (mu, extras) with extras = {"sigma": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]

    s0 = variables_dict["s0"]

    mu = (
            g[occupation] * dt
            - fs[occupation] * rad
            - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad_multiroom(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad, but the building is split into rooms each with
    their own indoor/outdoor delta-T and their own heat-loss (g) and solar-gain
    (fs) response, summed to reconstruct the whole-building heating consumption.
    fs varies by room because rooms have different orientation/window exposure
    even though the outdoor radiation `rad` they all see is the same signal.
    alpha stays shared (only occupation-indexed) since elec_consumption is a
    whole-building electrical measurement with no per-room signal.

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    :param x: (n_room + 3)-column 2D array. x[:, :-3] are the n_room per-room
        dt columns (tin_room - text), x[:, -3] is solar radiation, x[:, -2] is
        metered electrical consumption, x[:, -1] is the occupation flag.
    :param variables_dict: mandatory model variables are "g" and "fs" (shape
        (n_occ, n_room)) and "alpha"/"s0" (shape (n_occ,)).
    :return: (mu, extras) with extras = {"sigma": ...}.
    """
    dt = x[:, :-3]
    rad = x[:, -3]
    elec_consumption = x[:, -2]
    occupation = x[:, -1].astype(int)

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]

    s0 = variables_dict["s0"]

    heating_by_room = g[occupation] * dt - fs[occupation] * rad[:, None]
    mu = heating_by_room.sum(axis=1) - alpha[occupation] * elec_consumption

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad_lag(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad, plus a term on dt lagged by one day (dt_lag)
    to capture the "reheat" transient right after an unoccupied -> occupied
    setback recovery: the building must also recharge its thermal mass, on
    top of covering the current day's dt. h is expected negative: physically,
    h = -(C/dt_step) where C is the building's thermal capacitance -- a low
    dt_lag (coming out of an economy setback) combined with h<0 adds the
    extra "recharge" energy on top of g*dt, while on a stable day (dt_lag ~=
    dt) the h term nets out close to what g*dt alone already represents.

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    :param x: 5-column 2D array. x[:, 0] is dt, x[:, 1] is solar radiation,
        x[:, 2] is metered electrical consumption, x[:, 3] is the occupation
        flag, x[:, 4] is dt lagged by one day.
    :param variables_dict: "g", "fs", "alpha", "h", "s0" all shape (n_occ,).
    :return: (mu, extras) with extras = {"sigma": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)
    dt_lag = x[:, 4]

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]
    h = variables_dict["h"]
    s0 = variables_dict["s0"]

    mu = (
        g[occupation] * dt
        + h[occupation] * dt_lag
        - fs[occupation] * rad
        - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}

def heating_dt_occ_rad_DTdt(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad, plus a capacitive term c * DT_dt, the
    discretized C*dTint/dt term of the classical RC heat-balance equation
    (Q = UA*dt + C*dTint/dt - solar - internal gains): DT_dt is the (daily
    mean of the) centered numerical derivative of indoor temperature, and c
    is a single scalar capacity-like coefficient shared across occupation
    regimes, capturing the energy stored in/released from the building's
    thermal mass as the indoor temperature rises/falls, on top of the
    steady-state conduction loss g*dt.

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    :param x: 5-column 2D array. x[:, 0] is dt, x[:, 1] is solar radiation,
        x[:, 2] is metered electrical consumption, x[:, 3] is the occupation
        flag, x[:, 4] is DT_dt (dTint/dt).
    :param variables_dict: "g", "fs", "alpha", "s0" shape (n_occ,); "c" scalar.
    :return: (mu, extras) with extras = {"sigma": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)
    DT_dt = x[:, 4]

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]
    c = variables_dict["c"]
    s0 = variables_dict["s0"]

    mu = (
        g[occupation] * dt
        + c * DT_dt
        - fs[occupation] * rad
        - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad_DTdt_lags(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad_DTdt, but the capacitive term is a short
    distributed lag on DT_dt (today + 1..3 days back) instead of a single
    coefficient on today's value, to approximate a higher-order
    (multi-capacitance) thermal-mass response instead of a single-exponential
    one. The per-lag coefficients follow a geometric decay c_k = c0 * rho**k
    (k=0..3): c0 > 0 is today's capacity coefficient, rho in (0, 1) is the
    fraction of one day's capacitive contribution still present the next day
    (an implied memory half-life = ln(0.5) / ln(rho) days). Since c0 > 0 and
    0 < rho < 1, every c_k is guaranteed positive and monotonically
    decaying -- an earlier version built c_k as c0 plus a cumulative sum of
    unconstrained increments (a random walk), which let a higher lag's
    coefficient cross to negative in the fit, an unphysical sign flip for
    what should be a monotonically fading capacitive effect. The single
    shared decay rate also pools information across all 4 lags instead of
    estimating 3 independent step differences, which were poorly identified
    given how autocorrelated consecutive daily DT_dt values are.

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    :param x: 8-column 2D array. x[:, 0] is dt, x[:, 1] is solar radiation,
        x[:, 2] is metered electrical consumption, x[:, 3] is the occupation
        flag, x[:, 4] is DT_dt at lag 0 (today), x[:, 5:8] are DT_dt at lag
        1, 2, 3 days.
    :param variables_dict: "g", "fs", "alpha", "s0" shape (n_occ,); "c0"
        scalar (> 0); "rho" scalar in (0, 1) (geometric decay rate).
    :return: (mu, extras) with extras = {"sigma": ..., "lower": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)
    DT_dt = x[:, 4]
    DT_dt_lag1 = x[:, 5]
    DT_dt_lag2 = x[:, 6]
    DT_dt_lag3 = x[:, 7]

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]
    c0 = variables_dict["c0"]
    rho = variables_dict["rho"]
    s0 = variables_dict["s0"]

    c1 = c0 * rho
    c2 = c1 * rho
    c3 = c2 * rho
    capacity_term = c0 * DT_dt + c1 * DT_dt_lag1 + c2 * DT_dt_lag2 + c3 * DT_dt_lag3

    mu = (
        g[occupation] * dt
        + capacity_term
        - fs[occupation] * rad
        - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad_DTdt_wall_Ci(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad_DTdt_wall, but adds a second capacitive term on
    the indoor-air node itself: Ci * (Tin_23:00 - Tin_00:00), the intraday
    swing of indoor temperature (approximating Ci * integral(dTint/dt) over
    the day), in addition to the existing lumped-wall/mass term C * (beta*
    DTint_dt + (1-beta)*DText_dt). Ci is a single scalar shared across
    occupied/unoccupied regimes (unlike g/fs/alpha).

    :param x: 7-column 2D array. x[:, 0] is dt, x[:, 1] is solar radiation,
        x[:, 2] is metered electrical consumption, x[:, 3] is the occupation
        flag, x[:, 4] is DTint_dt (day-to-day drift of daily-mean Tint),
        x[:, 5] is DText_dt (day-to-day drift of daily-mean Text), x[:, 6] is
        DTint_intraday (today's Tin at end of day minus Tin at start of day).
    :param variables_dict: "g", "fs", "alpha", "s0" shape (n_occ,); "C",
        "beta", "Ci" scalars ("C" >= 0, "beta" in (0, 1), "Ci" >= 0).
    :return: (mu, extras) with extras = {"sigma": ..., "lower": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)
    DTint_dt = x[:, 4]
    DText_dt = x[:, 5]
    DTint_intraday = x[:, 6]

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]
    C = variables_dict["C"]
    beta = variables_dict["beta"]
    Ci = variables_dict["Ci"]
    s0 = variables_dict["s0"]

    wall_capacity_term = C * (beta * DTint_dt + (1 - beta) * DText_dt)
    air_capacity_term = Ci * DTint_intraday

    mu = (
        g[occupation] * dt
        + wall_capacity_term
        + air_capacity_term
        - fs[occupation] * rad
        - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def heating_dt_occ_rad_DTdt_wall_Ci_radlag(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad_DTdt_wall_Ci, plus a delayed solar gain term
    fm * rad_lag1: yesterday's solar radiation, stored in the thermal mass
    and released today, further reduces today's heating need. fm is
    per-occupation, like fs.

    :param x: 8-column 2D array. Columns 0-6 as in
        heating_dt_occ_rad_DTdt_wall_Ci, x[:, 7] is rad_lag1 (previous day's
        solar radiation).
    :param variables_dict: as heating_dt_occ_rad_DTdt_wall_Ci, plus "fm"
        shape (n_occ,) (>= 0).
    :return: (mu, extras) with extras = {"sigma": ..., "lower": ...}.
    """
    mu, extras = heating_dt_occ_rad_DTdt_wall_Ci(x[:, :7], variables_dict)
    occupation = x[:, 3].astype(int)
    rad_lag1 = x[:, 7]

    mu = mu - variables_dict["fm"][occupation] * rad_lag1

    return mu, extras


def heating_dt_occ_rad_DTdt_wall(x, variables_dict: dict):
    """
    Same as heating_dt_occ_rad_DTdt, but the capacitive term approximates a
    single lumped wall/thermal-mass node instead of using indoor temperature
    directly: the wall temperature is modeled as a quasi-static linear blend
    of the two known boundary temperatures, T_wall ~= beta*Tint + (1-beta)*Text,
    and the capacitive term is C * d(T_wall)/dt, discretized as the
    day-to-day drift of that blend: C * (beta*DTint_dt + (1-beta)*DText_dt),
    where DTint_dt/DText_dt are each the one-day-lagged difference of the
    daily-mean indoor/outdoor temperature (today's daily mean minus
    yesterday's). beta in (0, 1) locates the wall thermally: close to 1 means
    the modeled mass behaves like the indoor air (mass on the inside of the
    insulation), close to 0 means it tracks outdoor conditions (mass outside
    the insulation / lightly insulated envelope).

    The likelihood scale is a per-occupation noise floor: sigma = s0[occupation].
    "sigma" is a reserved key in the returned extras dict: PymcWrapper.build_model
    uses it in place of variables_dict["sigma"] whenever a model_function
    provides it (see its docstring).

    :param x: 6-column 2D array. x[:, 0] is dt, x[:, 1] is solar radiation,
        x[:, 2] is metered electrical consumption, x[:, 3] is the occupation
        flag, x[:, 4] is DTint_dt (day-to-day drift of daily-mean Tint),
        x[:, 5] is DText_dt (day-to-day drift of daily-mean Text).
    :param variables_dict: "g", "fs", "alpha", "s0" shape (n_occ,); "C"
        scalar (>= 0); "beta" scalar in (0, 1) (indoor/outdoor blend weight).
    :return: (mu, extras) with extras = {"sigma": ..., "lower": ...}.
    """
    dt = x[:, 0]
    rad = x[:, 1]
    elec_consumption = x[:, 2]
    occupation = x[:, 3].astype(int)
    DTint_dt = x[:, 4]
    DText_dt = x[:, 5]

    g = variables_dict["g"]
    fs = variables_dict["fs"]
    alpha = variables_dict["alpha"]
    C = variables_dict["C"]
    beta = variables_dict["beta"]
    s0 = variables_dict["s0"]

    capacity_term = C * (beta * DTint_dt + (1 - beta) * DText_dt)

    mu = (
        g[occupation] * dt
        + capacity_term
        - fs[occupation] * rad
        - alpha[occupation] * elec_consumption
    )

    sigma = s0[occupation]

    return mu, {"sigma": sigma, "lower": pt.constant(0.0)}


def ppv_projected_rad_cst_eff(x, variables_dict: dict):
    """
    Simplest model of pv panels. Constant efficiency.
    Radiation are provided as kWh/m² and must already be projected in the plan
    of the pannel
    :param x: The features . x[:, 0] is the solar radiation in Wh/m² or kWh/m²
    :param variables_dict: mandatory model variables are : "surface" and "efficiency"
    :return:  surface * efficiency * radiations
    """
    rad = x[:, 0]
    efficiency = variables_dict["efficiency"]
    surface = variables_dict["surface"]

    return surface * efficiency * rad, {"sigma": variables_dict["sigma"]}


def ppv_noct_model(x, variables_dict: dict):
    """Compute PV panel production

    :param x: 2d array
        - x[:, 0] : air_temperature [°C]
        - x[:, 1] : GTI [W/m2] normal to the panel
    :param variables_dict. Variables names are
    - peak_power: float: Peak power [Wp]
    - noct: float: Normal Operating Cell Temeprature [°C]
    - power_temp_coeff: float (positif): Efficiency loss by temperature ratio [%/°K]
    - inverter_eff: Inverter efficiency [-]

    returns panel_power [W]
    """

    air_temp = x[:, 0]
    rad = x[:, 1]

    noct = variables_dict["noct"]
    peak_power = variables_dict["peak_power"]
    power_temp_coeff = variables_dict["power_temp_coeff"]
    inverter_eff = variables_dict["inverter_eff"]

    # fmt: off
    panel_temp = (
            air_temp
            + rad * (noct - NOCT_AIR_TEMP) / NOCT_IRRADIANCE
    )

    panel_power = (
            peak_power
            * (1 - power_temp_coeff * (panel_temp - STC_AIR_TEMP))
            * rad / STC_IRRADIANCE
    )

    return inverter_eff * panel_power, {"sigma": variables_dict["sigma"]}
