"""Formula equivalents of the legacy functions of :mod:`bayesbuilding.models`.

Frozen translation table, used to migrate configs that name a legacy model
(``"model": "heating_dt_occ_rad"``) to a formula (``"model": {"mu": ...}``),
see :func:`legacy_to_formula`. ``tests/test_formula_legacy_equivalence.py``
checks each formula evaluates exactly like its legacy function. Do not add
entries: new models are written directly as formulas.

``LEGACY_INPUTS`` gives the column order each legacy function indexes via
``x[:, i]``.
"""

_SETBACK_RAD = (
    "switch(is_heating, base[0] + {g}*max({tau} - text, 0) - fs*rad, base[1])"
)
_DT_CORE = "g[occ]*dt - fs[occ]*rad - alpha[occ]*elec_consumption"
_WALL = "C*(beta*DTint_dt + (1 - beta)*DText_dt)"

LEGACY_FORMULAS: dict[str, dict] = {
    "season_cp_heating_es_dt": {"mu": "g*max(dt - tau, 0) + base", "sigma": "sigma"},
    "season_cp_heating_es": {"mu": "g*max(tau - text, 0) + base", "sigma": "sigma"},
    "heating_es_dju": {"mu": "g*dju + base", "sigma": "sigma"},
    "season_cp_occ_cp_heating_es": {
        "mu": "g[occ]*max(tau[occ] - text, 0) + baseline[occ]",
        "sigma": "sigma[occ]",
    },
    "season_cp_occ_cp_es_dt": {
        "mu": "g[occ]*max(dt - tau[occ], 0) + baseline[occ]",
        "sigma": "sigma[occ]",
    },
    "heating_es_dju_rad": {"mu": "base + g*dju - fs*rad", "sigma": "sigma"},
    "season_cp_heating_es_rad": {
        "mu": _SETBACK_RAD.format(g="g", tau="tau"),
        "sigma": "sigma",
    },
    "season_cp_heating_es_setback": {
        "mu": "switch(is_heating, base[0] + g*max(tau - text, 0), base[1])",
        "sigma": "sigma",
    },
    "heating_es_dju_rad_occ": {
        "mu": "base[occ] + g[occ]*dju - fs[occ]*rad",
        "sigma": "sigma",
    },
    "heating_es_dju_rad_occ_setback": {
        "mu": "switch(occ, base[0] + g*dju - fs*rad, base[1])",
        "sigma": "sigma",
    },
    "heating_cp_occ_rad": {
        "mu": "base[occ] + max(g[occ]*(tau[occ] - text) - fs[occ]*rad, 0)",
        "sigma": "sqrt(s0[occ]**2 + (s1*sigmoid((tau[occ] - text)/1.5))**2)",
    },
    "season_cp_heating_es_rad_g_by_period": {
        "mu": _SETBACK_RAD.format(g="g[period]", tau="tau"),
        "sigma": "sigma",
    },
    "season_cp_heating_es_rad_tau_by_period": {
        "mu": _SETBACK_RAD.format(g="g", tau="tau[period]"),
        "sigma": "sigma",
    },
    "season_cp_heating_es_rad_base_by_period": {
        "mu": (
            "switch(is_heating, base[is_heating, period] + g*max(tau - text, 0)"
            " - fs*rad, base[is_heating, period])"
        ),
        "sigma": "sigma",
    },
    "season_cp_heating_es_rad_g_tau_by_period": {
        "mu": _SETBACK_RAD.format(g="g[period]", tau="tau[period]"),
        "sigma": "sigma",
    },
    "heating_dt_occ_rad": {"mu": _DT_CORE, "sigma": "s0[occ]"},
    "heating_dt_occ_rad_multiroom": {
        "mu": (
            "g[occ, 0]*dt_chambre1 + g[occ, 1]*dt_cuisine + g[occ, 2]*dt_sejour"
            " - (fs[occ, 0] + fs[occ, 1] + fs[occ, 2])*rad"
            " - alpha[occ]*elec_consumption"
        ),
        "sigma": "s0[occ]",
    },
    "heating_dt_occ_rad_lag": {
        "mu": f"{_DT_CORE} + h[occ]*dt_lag",
        "sigma": "s0[occ]",
    },
    "heating_dt_occ_rad_DTdt": {"mu": f"{_DT_CORE} + c*DT_dt", "sigma": "s0[occ]"},
    "heating_dt_occ_rad_DTdt_lags": {
        "mu": (
            f"{_DT_CORE} + c0*DT_dt + c0*rho*DT_dt_lag1"
            " + c0*rho**2*DT_dt_lag2 + c0*rho**3*DT_dt_lag3"
        ),
        "sigma": "s0[occ]",
    },
    "heating_dt_occ_rad_DTdt_wall": {
        "mu": f"{_DT_CORE} + {_WALL}",
        "sigma": "s0[occ]",
    },
    "heating_dt_occ_rad_DTdt_wall_Ci": {
        "mu": f"{_DT_CORE} + {_WALL} + Ci*DTint_intraday",
        "sigma": "s0[occ]",
    },
    "heating_dt_occ_rad_DTdt_wall_Ci_radlag": {
        "mu": f"{_DT_CORE} + {_WALL} + Ci*DTint_intraday - fm[occ]*rad_lag1",
        "sigma": "s0[occ]",
    },
}

_DT_INPUTS = ("dt", "rad", "elec_consumption", "occ")

LEGACY_INPUTS: dict[str, tuple[str, ...]] = {
    "season_cp_heating_es_dt": ("dt",),
    "season_cp_heating_es": ("text",),
    "heating_es_dju": ("dju",),
    "season_cp_occ_cp_heating_es": ("occ", "text"),
    "season_cp_occ_cp_es_dt": ("occ", "dt"),
    "heating_es_dju_rad": ("dju", "rad"),
    "season_cp_heating_es_rad": ("text", "rad", "is_heating"),
    "season_cp_heating_es_setback": ("text", "is_heating"),
    "heating_es_dju_rad_occ": ("dju", "rad", "occ"),
    "heating_es_dju_rad_occ_setback": ("dju", "rad", "occ"),
    "heating_cp_occ_rad": ("text", "rad", "occ"),
    "season_cp_heating_es_rad_g_by_period": ("text", "rad", "is_heating", "period"),
    "season_cp_heating_es_rad_tau_by_period": ("text", "rad", "is_heating", "period"),
    "season_cp_heating_es_rad_base_by_period": ("text", "rad", "is_heating", "period"),
    "season_cp_heating_es_rad_g_tau_by_period": ("text", "rad", "is_heating", "period"),
    "heating_dt_occ_rad": _DT_INPUTS,
    "heating_dt_occ_rad_multiroom": (
        "dt_chambre1",
        "dt_cuisine",
        "dt_sejour",
        "rad",
        "elec_consumption",
        "occ",
    ),
    "heating_dt_occ_rad_lag": _DT_INPUTS + ("dt_lag",),
    "heating_dt_occ_rad_DTdt": _DT_INPUTS + ("DT_dt",),
    "heating_dt_occ_rad_DTdt_lags": _DT_INPUTS
    + ("DT_dt", "DT_dt_lag1", "DT_dt_lag2", "DT_dt_lag3"),
    "heating_dt_occ_rad_DTdt_wall": _DT_INPUTS + ("DTint_dt", "DText_dt"),
    "heating_dt_occ_rad_DTdt_wall_Ci": _DT_INPUTS
    + ("DTint_dt", "DText_dt", "DTint_intraday"),
    "heating_dt_occ_rad_DTdt_wall_Ci_radlag": _DT_INPUTS
    + ("DTint_dt", "DText_dt", "DTint_intraday", "rad_lag1"),
}


def legacy_to_formula(name: str) -> dict:
    """Formula spec (``{"mu", "sigma", "lower"}``) equivalent to legacy model
    ``name``."""
    try:
        return {**LEGACY_FORMULAS[name], "lower": 0.0}
    except KeyError:
        raise KeyError(
            f"No formula equivalent for legacy model {name!r}. "
            f"Available: {list(LEGACY_FORMULAS)}"
        )
