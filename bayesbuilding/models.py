import pymc as pm


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
