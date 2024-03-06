import pymc as pm


def occupation_changepoint(x, variables_dict: dict):
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
