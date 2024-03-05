import pymc as pm


def season_cp_heating_es(x, variable_dict):
    """
    Seasonal Change Point Heating Energy Signature

    During winter the overall building energy consumption is modeled as a linear
    function : Text * G + baseline. Where G is the overall heat loss [kWh/°C]
    and baseline is the "process" energy consumption.
    During summer, only baseline remains.
    The changepoint tau is based on the exterior temperature

    :param x: single column 2D array. x[:, 0] is the external air temperature
    :param variable_dict: variable dictionary
    :return: overall building consumption
    """
    t_ext = x[:, 0]
    g = variable_dict["g"]
    tau = variable_dict["tau"]
    baseline = variable_dict["base"]

    consumption = g * pm.math.maximum(tau - t_ext, 0)
    return consumption + baseline
