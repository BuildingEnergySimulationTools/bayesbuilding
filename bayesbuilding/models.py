import pymc as pm

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
    return set_point[wd_we]


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
    return baseline + heat + cool


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
    return consumption + baseline


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
    )


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
    )


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
    )


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
    )


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
    )


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
    )


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
    return consumption + baseline


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

    return baseline + g * dju - fs * rad


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

    return baseline[occ] + g[occ] * dju - fs[occ] * rad


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

    return pm.math.switch(occ, base[0] + g * dju - fs * rad, base[1])


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
    return consumption + baseline


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
    return consumption + baseline[occ]


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
    return consumption + baseline[occ]


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

    return pm.math.switch(x[:, 0], base_we, base_wd + fs * x[:, 1])


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
    return baseline + heat + cool + solar_cool + solar_heat

def heating_cp_occ_rad(x, variables_dict: dict):
    """
    comment
    """
    t_ext = x[:, 0]
    rad = x[:, 1]
    occupation = x[:, 2].astype(int)


    base = variables_dict["base"]
    g = variables_dict["g"]
    tau = variables_dict["tau"]
    fs = variables_dict["fs"]

    baseline = base[occupation]
    heat_minus_solar = g[occupation] * (tau[occupation] - t_ext) - fs[occupation] * rad

    return baseline + pm.math.maximum(heat_minus_solar, 0)

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

    return surface * efficiency * rad


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

    return inverter_eff * panel_power
